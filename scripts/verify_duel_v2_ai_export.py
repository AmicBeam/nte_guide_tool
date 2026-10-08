#!/usr/bin/env python3
"""Read-only numerical and action parity check for a CPU serving export."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    import numpy as np
    from app.modules.card_game.engine.ai.advanced_model import FrozenModel,encode
    from app.modules.card_game.rl.capability import LEGACY_MIRROR_KIND, PUBLIC_ASYMMETRIC_KIND
    results={}
    for deck in ('starter','weave-rush'):
        model=FrozenModel(args.directory,deck)
        kind=model.capability.get('kind')
        if kind not in (LEGACY_MIRROR_KIND, PUBLIC_ASYMMETRIC_KIND):
            raise SystemExit(f'unsupported capability {kind}')
        if kind==LEGACY_MIRROR_KIND and model.capability.get('human_public_custom'):
            raise SystemExit('legacy exact-mirror weights cannot claim public custom training')
        cases=json.loads((args.directory/f'{deck}-parity.json').read_text(encoding='utf-8'))
        largest=0.;elapsed=0.
        for case in cases:
            start=time.perf_counter()
            values,candidates=encode(case['state'],case['side'],case['actions'])
            np.testing.assert_array_equal(values,np.asarray(case['values'],dtype=np.float32))
            np.testing.assert_array_equal(candidates,np.asarray(case['candidates'],dtype=np.float32))
            scores=model.scores(values,candidates)
            elapsed+=time.perf_counter()-start
            error=float(np.max(np.abs(scores-np.asarray(case['scores']))));largest=max(largest,error)
            np.testing.assert_allclose(scores,case['scores'],rtol=2e-5,atol=2e-5)
            assert int(scores.argmax())==int(np.argmax(case['scores']))
        results[deck]=dict(cases=len(cases),max_score_error=largest,mean_encode_inference_ms=elapsed/len(cases)*1000)
    assert 'torch' not in sys.modules
    print(json.dumps(results));return 0
if __name__=='__main__':raise SystemExit(main())
