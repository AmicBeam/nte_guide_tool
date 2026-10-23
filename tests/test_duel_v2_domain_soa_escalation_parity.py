"""Native C++ CPU-oracle full-state parity for escalation recycle and deferred drain."""
from copy import deepcopy
import random
import shutil
import unittest
import torch

from app.modules.card_game.content.duel_v2 import CARDS
from app.modules.card_game.engine.duel_v2.flow import acting_side, apply_action, legal_actions
from app.modules.card_game.engine.duel_v2.state import shuffle
from app.modules.card_game.rl.batched_duel.domain_soa_codec import _decode_row
from app.modules.card_game.rl.batched_duel.domain_soa_execution import DomainSoAExecution
from app.modules.card_game.rl.batched_duel.domain_soa_lifecycle_source import supports_full_duel_rules
from app.modules.card_game.rl.batched_duel.domain_soa_schema import DomainSoABatch, DomainSoALayout
from tests.test_duel_v2_domain_soa_character_queries import pack, state


def without_presentation(st):
    st = deepcopy(st)
    st.pop("events", None)
    st.pop("_public_board", None)
    return st


def run_native(worlds, picks, sides=None):
    layout, fields = pack(worlds, layout=DomainSoALayout(game_capacity=len(worlds), max_characters=16))
    n = len(worlds)
    if sides is None:
        sides = [0 if st["active_side"] == "a" else 1 for st in worlds]
    sides = torch.as_tensor(sides, dtype=torch.int64)
    selected = torch.as_tensor(picks, dtype=torch.int64)
    active = torch.ones(n, dtype=torch.bool)
    with DomainSoAExecution(layout, device="cpu", allow_cpu_oracle=True, allow_unqualified=True) as core:
        out = core.launch(core.prepare(fields, sides, selected, active))
        batch = DomainSoABatch(layout=layout, **{k: v.numpy() for k, v in out["fields"].items()})
        return out, [_decode_row(batch, row) for row in range(n)]


def legal_index(st, action):
    actor = acting_side(st)
    legal = [item for item in legal_actions(st, actor) if item["type"] != "concede"]
    for index, item in enumerate(legal):
        if item == action:
            return index, actor
    raise AssertionError("action is not legal: %r among %r" % (action, legal[:8]))


def recycle_candidates(team):
    return [card for card in team["discard"]
            if card.get("card_id") in CARDS and not card.get("copy")
            and not card.get("derived") and not CARDS[card["card_id"]].get("derived")]


class DomainSoAEscalationParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (shutil.which("clang++") or shutil.which("g++")):
            raise unittest.SkipTest("Explicit C++ oracle compiler unavailable")

    def test_qualifications_stay_false(self):
        self.assertFalse(supports_full_duel_rules)
        self.assertFalse(DomainSoAExecution.supports_full_duel_rules)
        self.assertFalse(getattr(DomainSoAExecution, "gpu_qualified", False))

    def test_turn_twenty_recycle_uses_full_shuffle_stream(self):
        st = state(20261023, a="starter", b="starter")
        st["turn"] = 19
        st["escalation_enabled"] = True
        st["active_side"] = "a"
        st["phase"] = "playing"
        st["sides"]["a"]["ap"] = 0
        st["sides"]["a"]["normal_attack_available"] = False
        team = st["sides"]["b"]
        moved = []
        rest = []
        for card in team["deck"]:
            ok = card.get("card_id") in CARDS and not card.get("copy") and not card.get("derived") and not CARDS[card["card_id"]].get("derived")
            if ok and len(moved) < 5:
                moved.append(card)
            else:
                rest.append(card)
        team["deck"] = rest
        team["discard"] = list(team["discard"]) + moved
        self.assertGreaterEqual(len(recycle_candidates(team)), 4)
        expected = list(recycle_candidates(team))
        probe = {"rng": st["rng"]}
        shuffle(probe, expected)
        action = {"type": "end_turn"}
        index, actor = legal_index(st, action)
        gold = without_presentation(apply_action(st, actor, action))
        out, got = run_native([st], [index])
        self.assertEqual(int(out["errors"][0]), 0, out)
        self.assertIn(expected[0]["instance_id"], [card["instance_id"] for card in gold["sides"]["b"]["hand"]])
        self.assertNotIn(expected[0]["instance_id"], [card["instance_id"] for card in gold["sides"]["b"]["discard"]])
        self.assertEqual(got[0], gold)

    def test_drain_keeps_empty_deferred_list_present(self):
        st = state(77, a="starter", b="starter")
        cid = st["sides"]["a"]["order"][0]
        st["deferred_actions"] = [{"side": "a", "actor": cid, "entity_id": "e99999", "kind": "surplus"}]
        action = {"type": "end_turn"}
        index, actor = legal_index(st, action)
        gold = without_presentation(apply_action(st, actor, action))
        self.assertIn("deferred_actions", gold)
        self.assertEqual(gold["deferred_actions"], [])
        out, got = run_native([st], [index])
        self.assertEqual(int(out["errors"][0]), 0, out)
        self.assertIn("deferred_actions", got[0])
        self.assertEqual(got[0]["deferred_actions"], [])
        self.assertEqual(got[0], gold)

    def test_natural_collapse_game5_last_step_full_state(self):
        keys = ("starter", "weave-rush", "quick-rush", "zhenhong", "murk", "collapse", "lingke")
        row = 5
        world = state(900 + row, a=keys[row % 7], b=keys[(row + 1) % 7])
        rng = random.Random(1000 + row)
        for _ in range(47):
            actor = acting_side(world)
            legal = [item for item in legal_actions(world, actor) if item["type"] != "concede"]
            world = apply_action(world, actor, legal[rng.randrange(len(legal))])
            world["events"] = []
            world.pop("_public_board", None)
        actor = acting_side(world)
        legal = [item for item in legal_actions(world, actor) if item["type"] != "concede"]
        action = legal[rng.randrange(len(legal))]
        self.assertEqual(action["type"], "end_turn")
        gold = without_presentation(apply_action(world, actor, action))
        out, got = run_native([world], [legal.index(action)], sides=[0 if actor == "a" else 1])
        self.assertEqual(int(out["errors"][0]), 0, out)
        self.assertEqual(got[0].get("deferred_actions"), gold.get("deferred_actions"))
        self.assertEqual(got[0], gold)


if __name__ == "__main__":
    unittest.main()
