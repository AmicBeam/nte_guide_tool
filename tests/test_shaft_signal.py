"""Signal state is a configurable personal resource with timed display events."""
import json
import subprocess
import unittest
from pathlib import Path

from app.modules.shaft.domain.catalog import load_shaft_catalog
from app.modules.shaft.service import normalize_axis_payload, simulate_shaft_axis

ROOT = Path(__file__).resolve().parents[1]
CID = 'char_elyms'
RESOURCE = '信号等级'


class ShaftSignalTest(unittest.TestCase):
    def axis(self, level=0, steps=None, loop=True, nodes=()):
        return {'team': [dict(slot=0, character_id=CID, arc_id='', cartridge_id='',
                              awakening_nodes=list(nodes))],
                'steps': [dict(id=str(i), slot=0, action_id='action_elyms_'+name, start_tick=tick)
                          for i, (name, tick) in enumerate(steps or [('ultraskilla1',0),('melee1',20)])],
                'options': {'loop_enabled': loop, 'loop_initial_resources': {
                    CID: {'personal_resources': {RESOURCE: level}}}}}

    def result(self, axis):
        return simulate_shaft_axis(axis)['result']

    def test_initial_signal_selects_q_multiplier_and_resets_without_upgrade_rewards(self):
        for level, factor in [(0,1),(1,1.5),(2,2)]:
            with self.subTest(level=level):
                r = self.result(self.axis(level, nodes=[3]))
                self.assertEqual(r['resources_by_slot'][0]['initial_personal_resources'][RESOURCE], level)
                self.assertAlmostEqual(r['details'][0]['formula_parts']['base_multiplier_factor'], factor)
                self.assertEqual(r['details'][0]['personal_resources_after'][RESOURCE],0)
                self.assertFalse(any('当前形态' in w for w in r['details'][1]['warnings']))
                self.assertFalse(any(b['rule_id'] in ('character_elyms_illusion_c','character_elyms_illusion_b_c')
                                     for d in r['details'] for b in d['triggered_buffs']))

    def test_signal_transitions_and_b_upgrade_are_reflected_in_resources(self):
        for nodes, expected in [((),[1,2,0]),((2,),[1,2,0]),((1,2,3),[2,2,0])]:
            r = self.result(self.axis(0, [('skill2',0),('goodbye',10),('ultraskilla1',20)], loop=False, nodes=nodes))
            self.assertEqual([d['personal_resources_after'][RESOURCE] for d in r['details']], expected)
            self.assertEqual(r['resources_by_slot'][0]['personal_resources'][RESOURCE],0)
        r = self.result(self.axis(0, [('qte',0),('melee1',30)], loop=False))
        event = next(e for e in r['personal_resource_state_events'] if e['value']==1)
        self.assertEqual(event['tick'],r['details'][0]['end_tick'])
        self.assertGreater(event['tick'],r['details'][0]['start_tick'])

    def test_loop_initial_state_overrides_previous_tail_and_non_loop_ignores_loop_setting(self):
        r = self.result(self.axis(0,[('ultraskilla1',0),('skill2',100),('melee1',200)]))
        self.assertEqual(r['details'][0]['formula_parts']['base_multiplier_factor'],1)
        self.assertEqual(r['details'][1]['personal_resources_after'][RESOURCE],1)
        r = self.result(self.axis(2,loop=False))
        self.assertEqual(r['details'][0]['formula_parts']['base_multiplier_factor'],1)
        self.assertEqual(r['resources_by_slot'][0]['initial_personal_resources'][RESOURCE],0)

    def test_signal_normalization_is_integer_bounded_and_survives_roundtrip(self):
        for value, expected in [(-1,0),(1.9,1),(2,2),(100,2),('bad',0)]:
            axis=normalize_axis_payload(self.axis(value))
            stored=axis['options']['loop_initial_resources'][CID]['personal_resources'][RESOURCE]
            self.assertEqual(stored,expected)
            self.assertEqual(normalize_axis_payload(json.loads(json.dumps(axis)))['options'],axis['options'])
            r=self.result(self.axis(value))
            self.assertEqual(r['resources_by_slot'][0]['initial_personal_resources'][RESOURCE],expected)

    def test_frontend_resource_choices_and_weak_state_are_visible(self):
        source=(ROOT/'app/modules/shaft/static/js/shaft.js').read_text()
        def function(name):
            start=source.index('  function '+name+'(')
            end=source.index('\n  function ',start+1)
            return source[start:end]
        character=next(c for c in load_shaft_catalog()['characters'] if c['id']==CID)
        script='const character='+json.dumps(character,ensure_ascii=False)+';'
        script+='''const TIMELINE_FIXED_PERSONAL_RESOURCES={}, TIMELINE_HIDDEN_PERSONAL_RESOURCES=new Set();
          const state={catalog:{actions:[],formula_constants:{personal_resource_caps:{char_elyms:{信号等级:2}}}}};
          const getCharacterMap=()=>new Map([['char_elyms',character]]);
        '''+function('timelinePersonalResourceEntries')+function('loopPersonalResourceDefinitions')
        script+='console.log(JSON.stringify({defs:loopPersonalResourceDefinitions({character_id:"char_elyms"}),entries:timelinePersonalResourceEntries(character,{信号等级:0})}));'
        result=json.loads(subprocess.check_output(['node','-e',script],text=True))
        definition=result['defs'][0]
        self.assertEqual(definition['name'],RESOURCE)
        self.assertEqual(definition['max'],2)
        self.assertEqual([s['label'] for s in definition['states']],['弱信号','中等信号','强信号'])
        self.assertIn([RESOURCE,0],result['entries'])

    def test_cursor_uses_change_time_and_displays_initial_signal_on_empty_axis(self):
        source=(ROOT/'app/modules/shaft/static/js/shaft.js').read_text()
        start=source.index('  function slotResourcesAtCursor(')
        end=source.index('  function timelinePersonalResourceEntries(',start)
        script="""const character={id:'char_elyms',personal_resource_states:{信号等级:[]}};
          const getCharacterMap=()=>new Map([['char_elyms',character]]);
          const memberBySlot=()=>({character_id:'char_elyms'});
          const calculationTickFromVisual=v=>v;
          let result=null;
          const timelineResult=()=>result;
          const state={cursorTick:0,axis:{options:{loop_enabled:true,loop_initial_resources:{
            char_elyms:{personal_resources:{信号等级:2}}}}}};
        """+source[start:end]+"""
          const empty=slotResourcesAtCursor(0).personalResources.信号等级;
          state.axis.steps=[];
          result={resources_by_slot:[{slot:0,initial_personal_resources:{信号等级:0}}],personal_resource_state_events:[]};
          const emptyStale=slotResourcesAtCursor(0).personalResources.信号等级;
          state.axis.steps=[{}];
          result={resources_by_slot:[{slot:0,initial_personal_resources:{信号等级:0}}],
            details:[{start_tick:0,resources_after_by_slot:[{slot:0,personal_resources:{信号等级:1}}]}],
            personal_resource_state_events:[{tick:15,slot:0,resource:'信号等级',value:1}]};
          state.cursorTick=10;
          const before=slotResourcesAtCursor(0).personalResources.信号等级;
          state.cursorTick=15;
          const after=slotResourcesAtCursor(0).personalResources.信号等级;
          console.log(JSON.stringify({empty,emptyStale,before,after}));
        """
        result=json.loads(subprocess.check_output(['node','-e',script],text=True))
        self.assertEqual(result,dict(empty=2,emptyStale=2,before=0,after=1))

    def test_loop_confirmation_saves_enum_select_value(self):
        source=(ROOT/'app/modules/shaft/static/js/shaft.js').read_text()
        start=source.index('  function confirmLoopSettings(')
        end=source.index('\n  function ',start+1)
        script="""const state={axis:{options:{}}};
          const getCharacterMap=()=>new Map([['char_elyms',{energy_capacity:100,personal_resource_integer_resources:['信号等级']}]]);
          const select={dataset:{loopInitialPersonalResource:'信号等级'},value:'2',getAttribute:n=>n==='max'?'2':null};
          const row={dataset:{loopResourceCharacter:'char_elyms'},querySelector:()=>null,
            querySelectorAll:s=>s==='[data-loop-initial-personal-resource]'?[select]:[]};
          const $=id=>id==='shaft-loop-enabled'?{checked:true}:{querySelectorAll:()=>[row]};
          const pushUndoSnapshot=()=>{},closeLoopSettings=()=>{},renderEditorActions=()=>{},
            markSimulationStale=()=>{},renderTimeline=()=>{},scheduleSimulation=()=>{},persistAxisDraft=()=>{};
        """+source[start:end]+"""
          confirmLoopSettings();
          console.log(JSON.stringify(state.axis.options.loop_initial_resources.char_elyms.personal_resources));
        """
        result=json.loads(subprocess.check_output(['node','-e',script],text=True))
        self.assertEqual(result,{RESOURCE:2})
