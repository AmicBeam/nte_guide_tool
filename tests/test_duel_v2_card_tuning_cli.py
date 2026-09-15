"""CLI lifecycle for fixed-team card allocation. No formal training."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def _main(argv):
    from scripts.train_duel_v2_cards import main
    return main(argv)


class CardTuningCliGuardTest(unittest.TestCase):
    def _stderr(self, argv):
        buf=io.StringIO()
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(buf):
            _main(argv)
        self.assertNotEqual(raised.exception.code, 0)
        return buf.getvalue()

    def test_check_does_not_import_ml(self):
        import sys
        blocked={name:None for name in ("torch","torchaudio","torchvision")}
        popped={
            name: sys.modules.pop(name)
            for name in (
                "app.modules.card_game.rl.card_tuning",
                "torch",
                "torchaudio",
                "torchvision",
            )
            if name in sys.modules
        }
        try:
            with patch.dict(sys.modules,blocked,clear=False), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(_main(["--check","--team","starter"]),0)
            result=json.loads(output.getvalue())
            self.assertEqual(result["character_ids"],["nanali","iloy","zero","jiuyuan"])
            self.assertFalse(result["learns_characters"])
            self.assertFalse(result["training_started"])
            self.assertEqual(result["cards_per_character"],8)
            self.assertEqual(result["max_copies"],2)
            self.assertNotIn("app.modules.card_game.rl.card_tuning", sys.modules)
        finally:
            sys.modules.update(popped)

    def test_nonfinite_budgets_are_rejected(self):
        cases=(
            ["--check","--seconds","nan"],
            ["--check","--seconds","inf"],
            ["--check","--learning-rate","nan"],
            ["--check","--entropy-coef=inf"],
            ["--check","--entropy-coef=-inf"],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                self.assertIn("Invalid training budget or optimizer options", self._stderr(argv))

    def test_output_never_overwrites_old_run(self):
        with tempfile.TemporaryDirectory() as directory:
            old=Path(directory)/"old-run"
            old.mkdir()
            marker=old/"keep.txt"
            marker.write_text("keep",encoding="utf-8")
            err=self._stderr(["--propose","--resume",str(old/"missing.pt"),"--output",str(old),"--n","2"])
            self.assertIn("Use a new output directory", err)
            self.assertEqual(marker.read_text(encoding="utf-8"),"keep")
            self.assertEqual(list(old.iterdir()),[marker])
            err=self._stderr(['--train','--worker','--policy-checkpoint','missing.pt','--output',str(old)])
            self.assertIn('Use a new output directory',err)
            self.assertEqual(marker.read_text(encoding='utf-8'),'keep')

    def test_invalid_team_json_is_a_cli_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'team.json'
            for value in ('broken', '[{}, {}, {}, {}]', '{"character_ids":null}'):
                path.write_text(value)
                self.assertIn('error:',self._stderr(['--check','--team-json',str(path)]))


@unittest.skipUnless(importlib.util.find_spec("torch"),"Torch required for card-tuning CLI checkpoint checks")
class CardTuningCliCheckpointTest(unittest.TestCase):
    TEAM=["zero","bohe","iloy","baicang"]

    def _stderr(self, argv):
        buf=io.StringIO()
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(buf):
            _main(argv)
        self.assertNotEqual(raised.exception.code, 0)
        return buf.getvalue()

    def _checkpoint(self, root, **overrides):
        import torch
        from app.modules.card_game.rl.card_tuning import CardTuner
        from app.modules.card_game.rl.public_schema import rule_identity
        root=Path(root)
        root.mkdir(parents=True, exist_ok=True)
        tuner=CardTuner(self.TEAM,8)
        payload={**tuner.metadata(),"rule_hash":rule_identity(),"model":tuner.state_dict(),
                 "policy":{"sha256":"policy-a"},"opponents":[{"sha256":"opp-a"}]}
        payload.update(overrides)
        path=root/"latest.pt"
        torch.save(payload,path)
        return path,tuner

    def test_propose_keeps_exact_checkpoint_team(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ckpt,_=self._checkpoint(root)
            out=root/"proposals"
            self.assertEqual(_main(["--propose","--resume",str(ckpt),"--output",str(out),"--n","3","--seed","9"]),0)
            proposal=json.loads((out/"proposal.json").read_text(encoding="utf-8"))
            self.assertEqual(proposal["character_ids"],self.TEAM)
            self.assertFalse(proposal["learns_characters"])
            self.assertFalse(proposal["training_started"])
            self.assertFalse(proposal["independently_evaluated"])
            self.assertEqual(proposal["resume"],str(ckpt))
            self.assertEqual(proposal["checkpoint_sha256"],hashlib.sha256(ckpt.read_bytes()).hexdigest())
            self.assertEqual(proposal["schema"],"fixed_team_card_tuner_v1")
            self.assertTrue(proposal["gpu_lock"])
            self.assertTrue(proposal["rule_hash"])
            self.assertEqual(proposal["hidden"],8)
            self.assertEqual(proposal["candidates"],3)
            for index in range(1,4):
                deck=json.loads((out/f"candidate-{index:03d}.json").read_text(encoding="utf-8"))
                self.assertEqual(deck["character_ids"],self.TEAM)
                self.assertEqual(len(deck["card_ids"]),32)

    def test_resume_team_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ckpt,_=self._checkpoint(root)
            err=self._stderr(["--propose","--resume",str(ckpt),"--team","starter","--output",str(root/"new"),"--n","1"])
            self.assertIn("Resume cannot change the fixed team", err)
            self.assertFalse((root/"new").exists())

    def test_schema_and_gpu_identity_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            schema_ckpt,_=self._checkpoint(root/"schema", schema="other_schema")
            gpu_ckpt,_=self._checkpoint(root/"gpu", gpu_lock="not-the-catalog")
            rule_ckpt,_=self._checkpoint(root/"rules", rule_hash="deadbeef")
            cases=(
                (schema_ckpt,"Card-tuner checkpoint schema/rules mismatch"),
                (gpu_ckpt,"Card-tuner checkpoint schema/rules mismatch"),
                (rule_ckpt,"Card-tuner checkpoint schema/rules mismatch"),
            )
            for ckpt,needle in cases:
                with self.subTest(ckpt=str(ckpt)):
                    out=root/"out"/ckpt.parent.name
                    err=self._stderr(["--propose","--resume",str(ckpt),"--output",str(out),"--n","1"])
                    self.assertIn(needle, err)
                    self.assertFalse(out.exists())

    def test_battle_policy_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ckpt,_=self._checkpoint(root)
            policy=root/"policy.pt"; policy.write_bytes(b"policy-bytes")
            opponent=root/"opponent.pt"; opponent.write_bytes(b"opponent-bytes")
            err=self._stderr(["--train","--resume",str(ckpt),"--policy-checkpoint",str(policy),
                              "--opponent-checkpoint",str(opponent),"--output",str(root/"train-out"),"--n","1"])
            self.assertIn("Resume requires the same frozen battle policies", err)
            self.assertFalse((root/"train-out").exists())


if __name__=="__main__":
    unittest.main()
