"""Exe's cursor label follows rating thresholds and the actual resonance selection."""
import json
from pathlib import Path
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ExeRatingDisplayTest(unittest.TestCase):
    def test_score_labels_thresholds_resonances_and_other_resources(self):
        source=(ROOT/'app/modules/shaft/static/js/shaft.js').read_text()
        def function(name):
            start=source.index('  function '+name+'(')
            end=source.index('\n  function ',start+1)
            return source[start:end]
        character=next(c for c in json.loads((ROOT/'app/modules/shaft/static/data/characters.json').read_text())if c['id']=='char_exe')
        script='const character='+json.dumps(character,ensure_ascii=False)+';'
        script+='const formatNumber=(value,places)=>Number(value).toFixed(places);'
        for name in ['normalizeAwakeningNodes','normalizeAwakeningResonances','activeAwakeningCount','timelinePersonalResourceLabel']:
            script+=function(name)
        script+='''const label=(score,member={awakening_resonances:[]})=>timelinePersonalResourceLabel(character,member,'评分',score);
        console.log(JSON.stringify({
          thresholds:[0,99.5,100,299.5,300,499.5,500,2999.5,3000,3333,5000].map(s=>label(s)),
          omega:label(5000,{awakening_resonances:[6]}),
          explicitOff:label(5000,{awakening_nodes:[1,2,3,4,5,6],awakening_resonances:[]}),
          legacy:label(5000,{awakening:6}),
          frame:timelinePersonalResourceLabel(character,{},'帧数',12),
          signal:timelinePersonalResourceLabel({personal_resource_states:{信号等级:[{value:1,label:'中等信号'}]}},{},'信号等级',1)
        }));'''
        result=json.loads(subprocess.check_output(['node','-e',script],text=True))
        self.assertEqual(result['thresholds'],['评分0 (B)','评分99.5 (B)','评分100 (A)','评分299.5 (A)','评分300 (S)','评分499.5 (S)','评分500 (SS)','评分2999.5 (SS)','评分3000 (SSS)','评分3333 (SSS)','评分5000 (SSS)'])
        self.assertEqual(result['omega'],'评分5000 (Ω)')
        self.assertEqual(result['explicitOff'],'评分5000 (SSS)')
        self.assertEqual(result['legacy'],'评分5000 (Ω)')
        self.assertEqual(result['frame'],'帧数 12.0')
        self.assertEqual(result['signal'],'中等信号')
