"""Character substat analysis keeps the rest of the team at its actual build."""
import json
import subprocess
import unittest
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog

ROOT = Path(__file__).resolve().parents[1]


class ShaftCharacterSubstatsTest(unittest.TestCase):
    def test_each_candidate_matches_manual_simulation_with_teammates_unchanged(self):
        catalog = load_shaft_catalog()
        basic_ids = {a['character_id'] for a in catalog['actions']
                     if a['action_type'] == '普攻' and a.get('multipliers', {}).get('atk', 0) > 0}
        characters = [c for c in catalog['characters'] if c['id'] in basic_ids][:2]
        actions = [next(a for a in catalog['actions']
                        if a['character_id'] == c['id'] and a['action_type'] == '普攻'
                        and a.get('multipliers', {}).get('atk', 0) > 0)
                   for c in characters]
        axis = {
            'team': [dict(slot=slot, character_id=c['id'], character_name=c['name'],
                          arc_id='', cartridge_id='', substat_counts=counts)
                     for slot, c, counts in zip([1, 3], characters, [
                         {'atk_pct': 73, 'crit_rate': 11, 'flat_atk': 9, 'harmony_strength': 40},
                         {'all_dmg': 2, 'crit_dmg': 31, 'flat_hp': 7}])],
            'steps': [dict(id=str(i), slot=slot, action_id=a['id'], start_tick=i*30)
                      for i, (slot, a) in enumerate(zip([1, 3], actions))],
            'character_builds': {c['id']: {'substat_counts': {'atk_pct': 4}} for c in characters},
        }
        script = """
          const fs = require('fs');
          const engine = require('./app/modules/shaft/static/js/shaft_engine.js');
          const {axis, catalog} = JSON.parse(fs.readFileSync(0, 'utf8'));
          const before = JSON.stringify({axis, catalog});
          const main = ['all_dmg', 'crit_rate', 'crit_dmg', 'harmony_strength',
                        'stagger_strength', 'atk_pct', 'hp_pct', 'def_pct'];
          const cases = [1, 3].map(slot => {
            const analysis = engine.analyzeSubstatContributions(axis, catalog, {slot});
            const baseline = JSON.parse(JSON.stringify(axis));
            const member = baseline.team.find(m => m.slot === slot);
            member.substat_counts = Object.fromEntries(analysis.fields.map(f => [
              f.key, main.includes(f.key) ? 15 : 0]));
            const total = engine.simulateAxis(baseline, catalog).summary.total_damage;
            const expected = analysis.fields.map(field => {
              const candidate = JSON.parse(JSON.stringify(baseline));
              candidate.team.find(m => m.slot === slot).substat_counts[field.key] += 5;
              return {key: field.key, total: engine.simulateAxis(candidate, catalog).summary.total_damage};
            });
            return {slot, analysis, total, expected};
          });
          const rejected = [{}, {slot: 2}, {slot: 'invalid'}, {slot: 1.5}].map(options => {
            try { engine.analyzeSubstatContributions(axis, catalog, options); return false; }
            catch (e) { return e.message.includes('请选择队伍中的角色'); }
          });
          console.log(JSON.stringify({cases, rejected, unchanged: before === JSON.stringify({axis, catalog})}));
        """
        result = json.loads(subprocess.check_output(
            ['node', '-e', script], input=json.dumps({'axis': axis, 'catalog': catalog}),
            text=True, cwd=ROOT, timeout=30))
        self.assertTrue(result['unchanged'])
        self.assertTrue(all(result['rejected']))
        for case in result['cases']:
            analysis = case['analysis']
            self.assertEqual(analysis['slot'], case['slot'])
            self.assertAlmostEqual(analysis['baseline_total_damage'], case['total'])
            expected = {e['key']: e['total'] for e in case['expected']
                        if e['total'] - case['total'] > max(1e-8, case['total'] * 1e-12)}
            self.assertEqual({r['stat_key'] for r in analysis['rows']}, set(expected))
            for row in analysis['rows']:
                self.assertEqual(row['slot'], case['slot'])
                self.assertAlmostEqual(row['boosted_total_damage'], expected[row['stat_key']])
            self.assertEqual(analysis['rows'][0]['contribution_percent'], 100)

    def test_zero_damage_role_keeps_analysis_tabs_available(self):
        source = (ROOT / 'app/modules/shaft/static/js/shaft.js').read_text()
        def function(name):
            start = source.index('  function ' + name + '(')
            return source[start:source.index('\n  function ', start + 1)]
        script = """
          const content = {innerHTML: ''};
          const actionAnalysisState = {slot: 3, dimension: 'action', view: 'donut'};
          const $ = () => content;
          const actionContributionForSlot = () => ({actions: [], total_damage: 0});
          const actionContributionGroups = () => [];
          let calculated = null;
          const calculateCharacterSubstats = (slot, id) => { calculated = {slot, id}; };
        """ + function('actionAnalysisToolbar') + function('renderActionContributionAnalysis') + """
          renderActionContributionAnalysis();
          const empty = content.innerHTML;
          actionAnalysisState.dimension = 'substats';
          renderActionContributionAnalysis();
          console.log(JSON.stringify({empty, substats: content.innerHTML, calculated}));
        """
        result = json.loads(subprocess.check_output(['node', '-e', script], text=True))
        self.assertIn('data-analysis-dimension="substats"', result['empty'])
        self.assertIn('该角色当前没有造成动作伤害', result['empty'])
        self.assertIn('data-substat-analysis', result['substats'])
        self.assertEqual(result['calculated']['slot'], 3)
