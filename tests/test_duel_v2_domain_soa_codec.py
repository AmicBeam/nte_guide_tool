"""Domain SoA codec: seven-team mulligan/playing roundtrip and explicit rejects."""
from copy import deepcopy
import unittest

from app.modules.card_game.content.duel_v2 import STARTER_DECKS
from app.modules.card_game.engine.duel_v2.entities import CardEntity, CharacterEntity, PlayerEntity
from app.modules.card_game.engine.duel_v2.flow import acting_side, apply_action, legal_actions, new_game
from app.modules.card_game.rl.batched_duel.domain_soa_codec import (
    coverage,
    decode_domain_states,
    encode_domain_states,
    supports_full_duel_rules,
)
from app.modules.card_game.rl.batched_duel.domain_soa_schema import (
    CHARACTER_INT_FIELDS,
    DomainSoALayout,
    DomainSoAUnsupportedError,
    DomainSoADecodeError,
    PLAYER_INT_FIELDS,
    PRESENCE_ABSENT,
    PRESENCE_CATALOG,
    PRESENCE_NULL,
    PRESENCE_VALUE,
)

KEYS = ("starter", "weave-rush", "quick-rush", "zhenhong", "murk", "collapse", "lingke")
PRESENTATION = ("events", "_public_board")


def _decks():
    return {item["id"]: item for item in STARTER_DECKS}


def _fourteen_initial_states():
    decks = _decks()
    states = []
    for game in range(14):
        first = "a" if game % 2 == 0 else "b"
        pair = (game // 2) % 7
        states.append(
            new_game(
                seed=10160000 + game,
                decks={"a": decks[KEYS[pair]], "b": decks[KEYS[(pair + 1) % 7]]},
                first_side=first,
            )
        )
    return states


def _authoritative(state):
    return {key: value for key, value in state.items() if key not in PRESENTATION}


def _assert_same(left, right, path="root"):
    if type(left) is not type(right):
        raise AssertionError(f"{path}: {type(left).__name__} != {type(right).__name__}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path} key set changed {set(left) ^ set(right)}")
        for key in left:
            _assert_same(left[key], right[key], path + "." + str(key))
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{path} length changed")
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_same(a, b, path + f"[{index}]")
    elif left != right:
        raise AssertionError(f"{path}: {left!r} != {right!r}")


def _roundtrip(state, presentation="drop"):
    packed = encode_domain_states([state], DomainSoALayout(), presentation=presentation)
    got = decode_domain_states(packed)[0]
    _assert_same(_authoritative(state), got)
    return packed, got


def _natural_states(max_steps=40):
    decks = _decks()
    states = []
    for game in range(14):
        first = "a" if game % 2 == 0 else "b"
        pair = game % 7
        state = new_game(
            seed=10160000 + game,
            decks={"a": decks[KEYS[pair]], "b": decks[KEYS[(pair + 1) % 7]]},
            first_side=first,
            skip_mulligan=(game % 3 == 0),
        )
        states.append(state)
        steps = 0
        while state.get("phase") != "finished" and steps < max_steps:
            side = acting_side(state)
            acts = legal_actions(state, side)
            if not acts:
                break
            playable = [item for item in acts if item.get("type") not in ("end_turn", "concede")]
            pool = playable or [item for item in acts if item.get("type") != "concede"]
            state = apply_action(state, side, pool[steps % len(pool)])
            states.append(state)
            steps += 1
    return states

class DomainSoACodecTest(unittest.TestCase):
    def test_known_unimplemented_fields_and_mutable_aliases_are_rejected(self):
        for mutate in (
            lambda s: s["sides"]["a"].update(records=[{}]),
            lambda s: s["sides"]["a"].update(side=None),
            lambda s: s["sides"]["a"]["deck"].append(s["sides"]["a"]["hand"][0]),
            lambda s: s["sides"]["b"].update(entity_id=s["sides"]["a"]["entity_id"]),
        ):
            state = _fourteen_initial_states()[0]
            mutate(state)
            with self.assertRaises(DomainSoAUnsupportedError):
                encode_domain_states([state], presentation="drop")

    def test_catalog_binding_and_unused_column_corruption_fail_closed(self):
        from unittest.mock import patch
        from app.modules.card_game.content.duel_v2 import CARDS
        state = _fourteen_initial_states()[0]
        packed = encode_domain_states([state], presentation="drop")
        key = state["sides"]["a"]["hand"][0]["card_id"]
        with patch.dict(CARDS[key], cost=int(CARDS[key]["cost"]) + 1):
            with self.assertRaises(DomainSoADecodeError):
                decode_domain_states(packed)
        packed.effect_value_int[0, 0, 0] = 1
        with self.assertRaises(DomainSoADecodeError):
            decode_domain_states(packed)

    def test_torch_export_rejects_float_or_bool_row_indices(self):
        import torch
        packed = encode_domain_states([_fourteen_initial_states()[0]], presentation="drop").to_torch()
        for index in (torch.tensor([.9]), torch.tensor([True])):
            with self.assertRaises(TypeError):
                packed.clone_rows(index)
        clone = packed.clone_rows(torch.tensor([0], dtype=torch.int64))
        clone.arrays["character_hp"].zero_()
        self.assertTrue(packed.arrays["character_hp"].any())
        from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch
        host = DomainSoABatch.from_torch(packed)
        host.character_hp.fill(0)
        self.assertTrue(packed.arrays["character_hp"].any())

    def test_coverage_does_not_claim_full_public17(self):
        self.assertFalse(supports_full_duel_rules)
        report = coverage()
        self.assertFalse(report["supports_full_duel_rules"])
        self.assertFalse(report["complete_soa"])
        self.assertFalse(report["full_public17"])
        self.assertTrue(report["initial_mulligan"])
        self.assertTrue(report["normal_playing"])
        self.assertEqual(report["seven_official_presets"], KEYS)
        self.assertNotIn("Character.last_hit_source", report["missing_schema_fields"])
        self.assertNotIn("Effect.conditions payloads (count only)", report["missing_schema_fields"])

    def test_fourteen_seven_team_initial_mulligan_roundtrip(self):
        states = _fourteen_initial_states()
        packed = encode_domain_states(states, DomainSoALayout(), presentation="drop")
        restored = decode_domain_states(packed)
        self.assertEqual(packed.n_rows, 14)
        self.assertEqual(len(restored), 14)
        for original, got in zip(states, restored):
            self.assertEqual(original["phase"], "mulligan")
            _assert_same(_authoritative(original), got)
            self.assertIsInstance(got["sides"]["a"], PlayerEntity)
            hero = next(iter(got["sides"]["a"]["characters"].values()))
            self.assertIsInstance(hero, CharacterEntity)
            self.assertIsInstance(got["sides"]["a"]["hand"][0], CardEntity)

    def test_full_63bit_rng_is_python_int(self):
        states = _fourteen_initial_states()
        high = deepcopy(states[0])
        high["rng"] = (1 << 63) - 1
        packed = encode_domain_states(states + [high], DomainSoALayout(), presentation="drop")
        restored = decode_domain_states(packed)
        for original, got in zip(states, restored):
            self.assertIs(type(got["rng"]), int)
            self.assertEqual(got["rng"], original["rng"])
        self.assertEqual(restored[-1]["rng"], (1 << 63) - 1)
        self.assertGreaterEqual(max(state["rng"].bit_length() for state in states), 63)

    def test_normal_attack_seal_preserves_absence_null_and_runtime_reference(self):
        states=[]
        for value in (None,'','summon_e123'):
            st=_fourteen_initial_states()[0]
            st['sides']['b']['seal_normal_attack']=value
            states.append(st)
        states.append(_fourteen_initial_states()[0])
        encoded=encode_domain_states(states,presentation='drop')
        decoded=decode_domain_states(encoded)
        for expected,actual in zip(states,decoded):
            _assert_same(_authoritative(expected),actual)

    def test_entities_cards_and_zone_order(self):
        state = _fourteen_initial_states()[0]
        packed = encode_domain_states([state], presentation="drop")
        got = decode_domain_states(packed)[0]
        self.assertEqual(got["next_entity"], state["next_entity"])
        self.assertEqual(got["next_instance"], state["next_instance"])
        for side in ("a", "b"):
            self.assertEqual(got["sides"][side]["order"], state["sides"][side]["order"])
            self.assertEqual(
                [card["instance_id"] for card in got["sides"][side]["hand"]],
                [card["instance_id"] for card in state["sides"][side]["hand"]],
            )
            self.assertEqual(
                [card["entity_id"] for card in got["sides"][side]["deck"]],
                [card["entity_id"] for card in state["sides"][side]["deck"]],
            )
            self.assertEqual(got["sides"][side]["entity_id"], state["sides"][side]["entity_id"])

    def test_absent_null_zero_and_catalog_presence(self):
        state = _fourteen_initial_states()[0]
        packed = encode_domain_states([state], presentation="drop")
        fatigue = PLAYER_INT_FIELDS.index("fatigue")
        extra_ap = PLAYER_INT_FIELDS.index("extra_ap")
        attack = CHARACTER_INT_FIELDS.index("attack")
        harmony = CHARACTER_INT_FIELDS.index("harmony")
        self.assertEqual(int(packed.winner_presence[0]), PRESENCE_NULL)
        self.assertEqual(int(packed.player_front_presence[0, 0]), PRESENCE_NULL)
        self.assertEqual(int(packed.player_int_presence[0, 0, fatigue]), PRESENCE_VALUE)
        self.assertEqual(int(packed.player_int_value[0, 0, fatigue]), 0)
        self.assertEqual(int(packed.player_int_presence[0, 0, extra_ap]), PRESENCE_ABSENT)
        self.assertEqual(int(packed.character_int_presence[0, 0, 0, attack]), PRESENCE_CATALOG)
        self.assertEqual(int(packed.character_int_presence[0, 0, 0, harmony]), PRESENCE_VALUE)
        self.assertEqual(int(packed.character_int_value[0, 0, 0, harmony]), 0)
        self.assertEqual(int(packed.card_source_presence[0, 0]), PRESENCE_CATALOG)

    def test_rejects_incomplete_choice_and_unknown_or_schema_missing_fields(self):
        state = _authoritative(deepcopy(_fourteen_initial_states()[0]))
        with self.assertRaises(DomainSoAUnsupportedError):
            encode_domain_states([{**state, "pending_choice": {"kind": "discard"}}])
        supported = deepcopy(state)
        supported['pending_choice'] = dict(side='a',kind='kit',prompt='x',cards=[],deck_cards=True)
        _roundtrip(supported)
        with self.assertRaises(DomainSoAUnsupportedError):
            encode_domain_states([{**state, 'deferred_actions': [1]}])
        for key, value in (('has_harmonized', True),('return_after',False),('temp_revive',True),('gulang_item','vinyl')):
            flagged=deepcopy(state)
            next(iter(flagged['sides']['a']['characters'].values()))['flags']={key:value}
            _roundtrip(flagged)
        hit=deepcopy(state)
        next(iter(hit['sides']['a']['characters'].values()))['last_hit_source']='a:nanali'
        _roundtrip(hit)

    def test_playing_effects_flags_front_choice_and_operation_roundtrip(self):
        state = _authoritative(deepcopy(new_game(seed=3, first_side="a", skip_mulligan=True)))
        team = state["sides"]["a"]
        cid = team["order"][0]
        hero = team["characters"][cid]
        team["front"] = cid
        team["used"] = {"instant": True}
        team["harmonized"] = {cid: True}
        team["attacked_this_turn"] = [hero["entity_id"]]
        hero["flags"] = {"collapse_count": 2, "pact": True, "next_bonus": 1}
        hero["effects"] = [{
            "effect_id": "e2:effect:1",
            "definition_id": "attack.buff",
            "property": "attack.panel",
            "op": "add",
            "value": 1,
            "source_entity_id": hero["entity_id"],
            "source_key": "ability",
            "scope": "self",
            "kind": "persistent",
            "layer": 0,
            "label": "attack buff",
            "clear_on": ["down"],
            "visibility": "public",
            "show_marker": False,
        }]
        hero["equipment"] = {
            "card_id": team["hand"][0]["card_id"],
            "equipment_id": "e3:weapon:1",
            "source_card_entity_id": team["hand"][0]["entity_id"],
        }
        hand_card = team["hand"][0]
        state["pending_choice"] = {
            "side": "a",
            "kind": "discard",
            "prompt": "discard 1",
            "cards": [hand_card],
            "actor": cid,
        }
        packed, got = _roundtrip(state)
        self.assertIs(got["pending_choice"]["cards"][0], got["sides"]["a"]["hand"][0])
        self.assertEqual(got["sides"]["a"]["front"], cid)
        self.assertEqual(got["sides"]["a"]["characters"][cid]["flags"]["pact"], True)
        self.assertEqual(int(packed.pending_choice_present[0]), 1)
        playing = deepcopy(state)
        playing["pending_choice"] = None
        playing["operation"] = {
            "side": "a",
            "actor": cid,
            "card": playing["sides"]["a"]["hand"][0],
            "target_id": "b:" + playing["sides"]["b"]["order"][0],
            "participants": [],
            "harmony": None,
            "first_bonus": 0,
            "redeem_bonus": 0,
            "record_candidate": False,
            "card_mark": None,
            "option_id": None,
            "ap_paid": 1,
        }
        _, got_op = _roundtrip(playing)
        self.assertIs(got_op["operation"]["card"], got_op["sides"]["a"]["hand"][0])
        finished = deepcopy(playing)
        finished["operation"] = None
        finished["phase"] = "finished"
        finished["winner"] = "a"
        finished["reason"] = "concede"
        _, got_fin = _roundtrip(finished)
        self.assertEqual(got_fin["winner"], "a")
        self.assertEqual(got_fin["reason"], "concede")

    def test_presentation_default_rejects_and_drop_omits(self):
        state = _fourteen_initial_states()[0]
        with self.assertRaises(DomainSoAUnsupportedError):
            encode_domain_states([state])
        got = decode_domain_states(encode_domain_states([state], presentation="drop"))[0]
        self.assertNotIn("events", got)
        self.assertNotIn("_public_board", got)

    def test_rejects_bool_as_int_and_unknown_keys_but_encodes_playing(self):
        state = _authoritative(deepcopy(_fourteen_initial_states()[0]))
        bad_hp = deepcopy(state)
        bad_hp["sides"]["a"]["hp"] = True
        with self.assertRaises(TypeError):
            encode_domain_states([bad_hp])
        playing = new_game(seed=3, first_side="a", skip_mulligan=True)
        packed, got = _roundtrip(playing)
        self.assertEqual(got["phase"], "playing")
        self.assertEqual(int(packed.phase[0]), 1)
        mystery = dict(state)
        mystery["mystery"] = 1
        with self.assertRaises(DomainSoAUnsupportedError):
            encode_domain_states([mystery])

    def test_seven_preset_natural_playing_roundtrips_or_rejects_explicitly(self):
        accepted = 0
        rejected = []
        phases = set()
        for state in _natural_states():
            try:
                _, got = _roundtrip(state)
            except DomainSoAUnsupportedError as exc:
                rejected.append((state.get("phase"), str(exc)))
                continue
            phases.add(got["phase"])
            accepted += 1
            self.assertIsInstance(got["sides"]["a"], PlayerEntity)
            self.assertIs(type(got["rng"]), int)
        self.assertGreaterEqual(accepted, 14)
        self.assertIn("playing", phases)
        self.assertEqual(rejected, [], msg=str(rejected[:5]))

    def test_mimic_variants_and_integer_used_clocks_roundtrip(self):
        for mimic in ({},{'id':None},{'id':''},{'id':'adler'}):
            state=_fourteen_initial_states()[0]
            team=state['sides']['a'];cid=team['order'][0]
            team['characters'][cid]['flags']['mimic']=mimic
            team['used']={'instant':False,'canhong_dot_attack_turn:e1':12,'zero_int':0,'true_bool':True}
            _roundtrip(state)

    def test_revealed_ids_and_visible_deck_keep_absent_null_empty_and_order(self):
        for revealed in (None,[],['a-1','b-4','a-1']):
            state=_fourteen_initial_states()[0]
            state['sides']['a']['revealed_ids']=revealed
            state['sides']['a']['deck_visible_turn']=state['turn']
            _roundtrip(state)
        state['sides']['a']['revealed_ids']=['a-001']
        with self.assertRaises(DomainSoAUnsupportedError):
            encode_domain_states([state],DomainSoALayout(),presentation='drop')

    def test_effective_card_view_preserves_physical_alias_and_identity(self):
        from app.modules.card_game.engine.duel_v2.state import effective_card
        decks = _decks()
        state = new_game(seed=8, decks={'a':decks['zhenhong'],'b':decks['starter']}, skip_mulligan=True)
        team = state['sides']['a']
        team['characters']['zhenhong']['awakened'] = True
        raw = team['hand'][0]
        view = effective_card(state, 'a', raw)
        self.assertIs(view['_physical_card'], raw)
        state['operation'] = {'side':'a','actor':'zhenhong','card':view,'participants':[],
                              'harmony':None,'first_bonus':0,'redeem_bonus':0,'record_candidate':False}
        _, got = _roundtrip(state)
        self.assertIs(got['operation']['card']['_physical_card'], got['sides']['a']['hand'][0])
        view['entity_id'] = 'e99999'
        with self.assertRaises(DomainSoAUnsupportedError):
            encode_domain_states([state], DomainSoALayout(), presentation='drop')

    def test_pending_sortie_preserves_empty_and_zero_bonus(self):
        for spec in ({}, {'bonus':0}, {'bonus':2}):
            state = _fourteen_initial_states()[0]
            cid = state['sides']['a']['order'][0]
            state['operation'] = {'side':'a','actor':cid,'card':None,'participants':[], 'harmony':None,
                                  'first_bonus':0,'redeem_bonus':0,'record_candidate':False,
                                  'pending_sortie':{'spec':spec,'stay_bench':False}}
            _roundtrip(state)
        state['operation']['pending_sortie']['spec']['future_unknown'] = True
        with self.assertRaises(DomainSoAUnsupportedError):
            encode_domain_states([state], DomainSoALayout(), presentation='drop')

    def test_effect_conditions_and_empty_clear_events_preserve_types(self):
        from app.modules.card_game.engine.duel_v2.modifiers import contribution
        for conditions in ([], [{'id':'enemy_turn'}], [{'id':'damage_attribute','args':{'attribute':'光'}}],
                           [{'id':'resource_at_least','args':{'resource':'energy','amount':2}}],
                           [{'id':'source_shape','args':{'card_id':'N08'}}], [{'id':'source_alive','args':{}}]):
            state = _fourteen_initial_states()[0]
            team = state['sides']['a']
            hero = team['characters'][team['order'][0]]
            effect = contribution('test.condition', 'attack.panel', 1, source_entity_id=hero['entity_id'], clear_on=[])
            effect['conditions'] = conditions
            hero['effects'] = [effect]
            _, got = _roundtrip(state)
            self.assertEqual(got['sides']['a']['characters'][team['order'][0]]['effects'][0]['conditions'], conditions)
            hero['effects'][0]['clear_on'] = ['ultimate_end']
            _roundtrip(state)
        state['sides']['a']['characters'][team['order'][0]]['effects'][0]['conditions'] = [{'id':'future_unknown'}]
        with self.assertRaises(DomainSoAUnsupportedError):
            encode_domain_states([state], DomainSoALayout(), presentation='drop')

    def test_runtime_fields_keep_null_false_zero_and_structured_chain(self):
        state=_fourteen_initial_states()[0]
        team=state['sides']['a'];cid=team['order'][0];other=team['order'][1]
        hero=team['characters'][cid]
        hero['collapse_reductions']={team['characters'][other]['entity_id']:2}
        team.update(effect_seq=2,support_turn=0,support_count=3)
        hero.update(last_hit_source='b:canhong',beast_fangs=3,base_attack=2,base_max_hp=4,
                    lingke_chain=dict(entity_id=team['characters'][other]['entity_id'],character_id=other,count=0))
        hero['flags'].update(has_harmonized=False,return_after=None,temp_revive=True,
                             gulang_item='vinyl',free_normal_attack_turn=0)
        _, got=_roundtrip(state)
        self.assertIs(got['sides']['a']['characters'][cid]['flags']['has_harmonized'],False)
        for value in (None,[]):
            variant=deepcopy(state);variant['deferred_actions']=value
            _,got=_roundtrip(variant)
            self.assertIs(type(got['deferred_actions']),type(value))
        chain_null=deepcopy(state);chain_null['sides']['a']['characters'][cid]['lingke_chain']=None
        _roundtrip(chain_null)
        bad=deepcopy(state);bad['sides']['a']['characters'][cid]['lingke_chain']['count']=False
        with self.assertRaises(TypeError):encode_domain_states([bad],presentation='drop')

    def test_storage_order_is_independent_of_lineup_and_keeps_dismissed_summon(self):
        from app.modules.card_game.engine.duel_v2.summons import summon_front,dismiss
        state=new_game(seed=63,skip_mulligan=True)
        team=state['sides']['a'];original=list(team['characters'])
        team['order']=[original[1],original[0],*original[2:]]
        team['attacked_this_turn']=[team['characters'][original[3]]['entity_id'],team['characters'][original[0]]['entity_id']]
        ghost=summon_front(state,'a',name='test summon',attack=2,hp=4)
        dismiss(state,'a',ghost)
        _,got=_roundtrip(state)
        self.assertEqual(list(got['sides']['a']['characters']),list(team['characters']))
        self.assertEqual(got['sides']['a']['order'],team['order'])
        self.assertIn(ghost,got['sides']['a']['characters'])
        self.assertNotIn(ghost,got['sides']['a']['order'])
        self.assertEqual(got['sides']['a']['attacked_this_turn'],team['attacked_this_turn'])
        team['order']=[]
        _,got=_roundtrip(state)
        self.assertEqual(got['sides']['a']['order'],[])
        self.assertEqual(list(got['sides']['a']['characters']),list(team['characters']))

    def test_encode_does_not_mutate_and_empty_batch_roundtrips(self):
        state = _fourteen_initial_states()[0]
        before = deepcopy(_authoritative(state))
        hand0 = state["sides"]["a"]["hand"][0]
        packed = encode_domain_states([state], presentation="drop")
        _assert_same(before, _authoritative(state))
        got = decode_domain_states(packed)[0]
        self.assertIsNot(got, state)
        self.assertIsNot(got["sides"]["a"]["hand"][0], hand0)
        self.assertIsNot(got["sides"]["a"]["hand"][0]["source"], hand0["source"])
        empty = encode_domain_states([])
        self.assertEqual(empty.n_rows, 0)
        self.assertEqual(decode_domain_states(empty), [])
