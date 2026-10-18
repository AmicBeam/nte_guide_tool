"""Pure planning/reference checks only; no rules, CUDA, rollout or qualification."""
import ast
import json
import inspect
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import unittest

from app.modules.card_game.content.duel_v2 import STARTER_DECKS, CARDS
from app.modules.card_game.rl.build_acceptance import canonical_build
from app.modules.card_game.rl.seven_schedule import ROSTER
from app.modules.card_game.rl.seven_auxiliary_jobs import AUXILIARY_DECISION_LIMIT, auxiliary_jobs, adapt_auxiliary_jobs
from app.modules.card_game.rl import seven_covered_value


def starting_decks():
    all_decks={d['id']:d for d in STARTER_DECKS}
    return {key:canonical_build(all_decks[key]) for key in ROSTER}


def plan_digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


class AuxiliaryJobsTest(unittest.TestCase):
    def test_plan_goldens_all_pairs_replicas_and_card_order(self):
        decks=starting_decks();before=deepcopy(decks)
        expected_digests={1: 'ee381fa9c9091f7d81e549320b9966231dc4b7b427bd721ab98ace7a13a60d83', 2: 'aff2143595b01cd1c3984da0b8e2f975fa55fada8c984c9dc1b5bb6b68fc92a5', 3: 'ed42ed4dbaa4f1fc6a31ff062262183dc39d99c68c9b792334932f2873556b0f', 6: '8d5dc8eef189bc16ef41e65ce1243e50ea97ecb052802139af7c3954c482d1e7', 9: 'cbaae750d73573afc12888f576d7ebf8793b3dd69f10cbe27699a26f82ebf171', 12: 'd397e074dec713333977acc5f4da433c82df4ee845aa95e8835a1af69c32eec1'}
        for replicas in expected_digests:
            with self.subTest(replicas=replicas):
                actual=auxiliary_jobs(123456,replicas,decks=decks,deadline=987654321.)
                self.assertEqual(plan_digest(actual),expected_digests[replicas])
                self.assertEqual(len(actual),56*replicas)
                self.assertEqual({key for j in actual for key in j['keys'].values()},set(ROSTER))
                for replica in range(replicas):
                    rows=actual[replica*56:(replica+1)*56]
                    for pair in range(28):
                        a,b=rows[2*pair:2*pair+2]
                        self.assertEqual(a['seed'],123456+replica*100+pair)
                        self.assertEqual(a['seed'],b['seed']);self.assertEqual((a['first'],b['first']),('a','b'))
                        self.assertEqual(a['decks'],b['decks'])
                        for job in (a,b):
                            for side,key in job['keys'].items():
                                deck=job['decks'][side]
                                self.assertEqual(deck['character_ids'],decks[key]['character_ids'])
                                self.assertEqual(deck['card_ids'],sorted(deck['card_ids']))
                                for hero in deck['character_ids']:
                                    self.assertEqual(sum(CARDS[c]['character_id']==hero for c in deck['card_ids']),8)
                                self.assertTrue(all(not CARDS[c].get('derived')for c in deck['card_ids']))
        self.assertEqual(decks,before)

    def test_explicit_frozen_start_and_order_never_replaced(self):
        decks=starting_decks()
        for d in decks.values():
            d['character_ids'].reverse();d['card_ids'].reverse();d['name']='user fixed start'
        before=deepcopy(decks)
        actual=auxiliary_jobs(9,2,decks=decks,deadline=200.)
        self.assertEqual(plan_digest(actual),'42f90cf458770eb46d73feb29f143a034955ff732e10888137b683648fc6a45b');self.assertEqual(decks,before)
        for job in actual:
            for side,key in job['keys'].items():
                self.assertEqual(job['decks'][side]['character_ids'],decks[key]['character_ids'])
                self.assertEqual(job['decks'][side]['name'],'user fixed start')

    def test_metadata_preserves_varied_build_identity_and_normal_mulligan(self):
        jobs=auxiliary_jobs(7,1,decks=starting_decks(),deadline=900.)
        # Metadata must preserve even unsorted actual initial card order.
        jobs[0]['decks']['a']['card_ids'].reverse();before=deepcopy(jobs)
        bound=adapt_auxiliary_jobs(jobs,starting_decks=starting_decks())
        self.assertEqual(list(bound.jobs),before);self.assertEqual(list(bound.identities),before)
        self.assertFalse(bound.gpu_qualified);self.assertFalse(bound.formal_training_qualified)
        self.assertEqual(bound.execution['exploration_steps'],6);self.assertEqual(bound.execution['value_start_step'],6)
        self.assertFalse(bound.execution['counts_as_search'])
        self.assertEqual(bound.execution['max_decisions'],AUXILIARY_DECISION_LIMIT)
        self.assertEqual(AUXILIARY_DECISION_LIMIT,512)
        for job,init,keys in zip(jobs,bound.initializers,bound.keys):
            self.assertEqual(init,dict(seed=job['seed'],first_side=job['first'],decks=job['decks'],skip_mulligan=False,escalation=True))
            self.assertEqual(keys,job['keys'])
        bound.jobs[0]['decks']['a']['card_ids'].reverse()
        self.assertEqual(jobs,before);self.assertEqual(bound.initializers[0]['decks'],before[0]['decks'])
        self.assertEqual(bound.identities[0],before[0])

    def test_invalid_planner_parameters_and_schema(self):
        decks=starting_decks()
        for base,replicas,deadline in ((True,1,99),(-1,1,99),(2**63-1,2,99),(0,True,99),(0,4,99),(0,1,True),(0,1,float('inf')),(0,1,0)):
            with self.subTest(values=(base,replicas,deadline)),self.assertRaises(ValueError):
                auxiliary_jobs(base,replicas,decks=decks,deadline=deadline)
        for bad in ({},dict(decks,unknown=decks['starter'])):
            with self.assertRaises(ValueError):auxiliary_jobs(0,1,decks=bad,deadline=99)
        derived=next(k for k,v in CARDS.items()if v.get('derived'))
        bad=deepcopy(decks);bad['starter']['card_ids'][0]=derived
        with self.assertRaises(ValueError):auxiliary_jobs(0,1,decks=bad,deadline=99)

    def test_metadata_rejects_unknown_fields_keys_seed_deadline_and_duplicates(self):
        original=auxiliary_jobs(7,1,decks=starting_decks(),deadline=900.)[0]
        changes=[dict(seed=True),dict(seed=-1),dict(seed=2**63),dict(first='c'),dict(deadline=None),dict(deadline=True),
                 dict(deadline=float('nan')),dict(deadline=10**1000),dict(keys={'a':'unknown','b':'starter'}),dict(keys={'a':'starter'}),
                 dict(pure=True),dict(skip_mulligan=True)]
        for changeset in changes:
            with self.subTest(changes=changeset),self.assertRaises(ValueError):
                adapt_auxiliary_jobs([dict(deepcopy(original),**changeset)],starting_decks=starting_decks())
        for jobs in ([],[original,deepcopy(original)]):
            with self.assertRaises(ValueError):adapt_auxiliary_jobs(jobs,starting_decks=starting_decks())
        derived=next(k for k,v in CARDS.items()if v.get('derived'))
        bad=deepcopy(original);bad['decks']['a']['card_ids'][0]=derived
        with self.assertRaises(ValueError):adapt_auxiliary_jobs([bad],starting_decks=starting_decks())
        bad=deepcopy(original);bad['decks']['a']['unsupported']=True
        with self.assertRaises(ValueError):adapt_auxiliary_jobs([bad],starting_decks=starting_decks())

    def test_frozen_start_binding_rejects_wrong_team_or_order_and_is_independent(self):
        starts=starting_decks();jobs=auxiliary_jobs(19,1,decks=starts,deadline=1000.)
        # Protocol variations are valid even if names/IDs differ from starts.
        job=deepcopy(jobs[2]);job['decks']['a'].update(id='manual-build',name='manual name')
        job['decks']['a']['card_ids'].reverse()
        before=deepcopy(job);bound=adapt_auxiliary_jobs([job],starting_decks=starts)
        self.assertEqual(bound.jobs[0],before)
        self.assertEqual(bound.starting_decks,starts)
        self.assertEqual(bound.starting_lineups,{k:tuple(v['character_ids'])for k,v in starts.items()})
        starts['starter']['character_ids'].reverse();starts['starter']['card_ids'].reverse()
        self.assertNotEqual(bound.starting_decks,starts)
        self.assertEqual(bound.starting_lineups['starter'],tuple(starting_decks()['starter']['character_ids']))
        wrong=deepcopy(jobs[0]);wrong['decks']['a']=deepcopy(starting_decks()['lingke'])
        with self.assertRaises(ValueError):adapt_auxiliary_jobs([wrong],starting_decks=starting_decks())
        wrong=deepcopy(jobs[0]);wrong['decks']['a']['character_ids'].reverse()
        with self.assertRaises(ValueError):adapt_auxiliary_jobs([wrong],starting_decks=starting_decks())
        with self.assertRaises(TypeError):adapt_auxiliary_jobs([job])
        for bad in (None,{},dict(starting_decks(),unknown=starting_decks()['starter'])):
            with self.assertRaises(ValueError):adapt_auxiliary_jobs([job],starting_decks=bad)
        # A manually frozen reordered lineup is legal only when explicitly bound.
        starts=starting_decks()
        for deck in starts.values():deck['character_ids'].reverse()
        planned=auxiliary_jobs(19,1,decks=starts,deadline=1000.)
        self.assertEqual(list(adapt_auxiliary_jobs(planned,starting_decks=starts).jobs),planned)

    def test_cli_delegates_and_scope_identity_binds_actual_planner(self):
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'scripts/prepare_duel_v2_seven_value.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
        jobs=next(n for n in main.body if isinstance(n,ast.FunctionDef)and n.name=='jobs')
        self.assertEqual(len(jobs.body),1);call=jobs.body[0].value
        self.assertIsInstance(call,ast.Call);self.assertEqual(call.func.id,'auxiliary_jobs')
        self.assertEqual(ast.dump(call.keywords[1].value),ast.dump(ast.parse('end-30',mode='eval').body))
        folder=Path(seven_covered_value.__file__).parent;digest=sha256()
        names=('seven_covered_value.py','covered_rollout.py','seven_sampler.py','policies.py','seven_auxiliary_jobs.py')
        for name in names:digest.update((folder/name).read_bytes())
        self.assertEqual(seven_covered_value.identity(),digest.hexdigest())
        collector=ast.parse(inspect.getsource(seven_covered_value.collect_game))
        loops=[n for n in ast.walk(collector)if isinstance(n,ast.For)]
        physical=next(n for n in loops if isinstance(n.target,ast.Name)and n.target.id=='step')
        self.assertEqual(physical.iter.args[0].id,'AUXILIARY_DECISION_LIMIT')
        legacy=sha256()
        for name in names[:-1]:legacy.update((folder/name).read_bytes())
        self.assertNotEqual(seven_covered_value.identity(),legacy.hexdigest())

if __name__=='__main__':unittest.main()
