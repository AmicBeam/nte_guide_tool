"""CPU same-source template lifecycle oracle; not CUDA/search evidence."""
from copy import deepcopy
import unittest
from unittest.mock import patch


class PreparedTemplateReferenceTest(unittest.TestCase):
    def test_seven_teams_early_and_normal_candidates_match_independent_previews(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.content.duel_v2 import get_catalog
        from app.modules.card_game.rl.batched_duel.gpu_teacher_preview_parts import teacher_prepare,teacher_preview,teacher_finish
        from app.modules.card_game.rl.batched_duel.gpu_teacher_prepared_template import teacher_prepare_with_template,teacher_preview_with_template
        from app.modules.card_game.engine.duel_v2.projection import _PREPARED_PREVIEW
        builds=get_catalog()['starter_decks'];self.assertEqual(len(builds),7);jobs=0
        for i,build in enumerate(builds):
            for skip in (False,True):
                state=new_game(seed=20261014001+2*i+int(skip),decks={'a':build,'b':builds[(i+1)%7]},first_side='a',skip_mulligan=skip)
                before=deepcopy(state);view,legal,indices,template=teacher_prepare_with_template(state,'a')
                expected_view,expected_legal,expected_indices=teacher_prepare(state,'a')
                self.assertEqual((view,legal,indices),(expected_view,expected_legal,expected_indices))
                original_template=deepcopy(template);values=[];expected=[]
                for index in indices:
                    action=view['legal_actions'][index]['action'];packet={'template':template,'action':action}
                    # A correctly bound prepared context must never anonymize
                    # again, and every candidate must retain its own mutation.
                    with patch('app.modules.card_game.engine.duel_v2.projection._prepare_preview_template',side_effect=AssertionError('Repeated anonymization')):
                        values.append(teacher_preview_with_template(state,'a',packet))
                    expected.append(teacher_preview(state,'a',action));jobs+=1
                    self.assertEqual(template,original_template);self.assertEqual(state,before)
                    self.assertIsNone(_PREPARED_PREVIEW.get())
                self.assertEqual(values,expected)
                self.assertEqual(teacher_finish(view,legal,indices,values),teacher_finish(expected_view,expected_legal,expected_indices,expected))
                if not indices:self.assertIsNone(template)
        self.assertGreater(jobs,0)

    def test_stale_template_is_rejected_and_outer_context_is_restored(self):
        from app.modules.card_game.engine.duel_v2 import new_game
        from app.modules.card_game.rl.batched_duel.gpu_teacher_prepared_template import teacher_prepare_with_template,teacher_preview_with_template
        from app.modules.card_game.engine.duel_v2.projection import _PREPARED_PREVIEW
        state=new_game(seed=20261014999,first_side='a',skip_mulligan=True)
        view,legal,indices,template=teacher_prepare_with_template(state,'a')
        self.assertTrue(indices)
        packet={'template':template,'action':view['legal_actions'][indices[0]]['action']}
        previous={'outer':True};token=_PREPARED_PREVIEW.set(previous)
        try:
            teacher_preview_with_template(state,'a',packet)
            self.assertIs(_PREPARED_PREVIEW.get(),previous)
            stale=deepcopy(template);stale['version']+=1
            with self.assertRaises(ValueError):teacher_preview_with_template(state,'a',dict(packet,template=stale))
            self.assertIs(_PREPARED_PREVIEW.get(),previous)
        finally:_PREPARED_PREVIEW.reset(token)


if __name__=='__main__':unittest.main()
