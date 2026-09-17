#!/usr/bin/env python3
"""On-demand tactical policy/engine diagnostics, no optimization."""
import argparse,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--transfer-weights',action='store_true',help='Explicit diagnostic-only transfer across rule versions');a=p.parse_args()
    if a.output.exists():raise ValueError('Choose a new diagnostic output')
    from app.modules.card_game.rl.card_tuning import load_battle_policy
    from app.modules.card_game.rl.gpu_duel.compiled_backend import CompiledStarterBackend
    from app.modules.card_game.rl.strategy_quality import diagnose_network
    if a.transfer_weights:
        import torch,hashlib
        from app.modules.card_game.rl.gpu_duel.ppo import CompactScorer
        from app.modules.card_game.rl.public_training import warm_start_weights
        from app.modules.card_game.rl.public_schema import state_dim,rule_identity
        from app.modules.card_game.rl.opening_observation import OPENING_SCHEMA,OPENING_CAND_DIM
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        net=CompactScorer(state_dim(),OPENING_CAND_DIM,saved['hidden']);warm_start_weights(saved,net,saved['deck']);net.eval()
        identity={'sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'source_rules':saved['rule_identity'],'runtime_rules':rule_identity(OPENING_SCHEMA),'diagnostic_transfer_only':True}
    else:net,identity=load_battle_policy(a.checkpoint,'cpu')
    with tempfile.TemporaryDirectory() as directory:
        backend=CompiledStarterBackend(directory,backend='native')
        try:report=diagnose_network(net,backend=backend)
        finally:backend.close()
    report['model']=identity;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(report['cases']),'parity':True,'output':str(a.output)}))
if __name__=='__main__':main()
