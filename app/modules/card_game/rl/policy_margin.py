"""Summaries of how close the best card is to the best non-card action."""
import numpy as np


def play_margin(scores, kinds, play_kind):
    """Best legal card logit minus best legal non-card logit.

    Positive means a card is strictly ahead. None when either side is absent.
    """
    scores = np.asarray(scores, dtype=np.float64)
    kinds = np.asarray(kinds)
    play = scores[kinds == play_kind]
    other = scores[kinds != play_kind]
    if play.size == 0 or other.size == 0:
        return None
    return float(play.max() - other.max())


def margin_bins(margins):
    """Count how many best-card margins sit in a narrow band around zero."""
    values = np.asarray(list(margins), dtype=np.float64)
    ordered = [
        ('< -0.05', values < -0.05),
        ('[-0.05, -0.01)', (values >= -0.05) & (values < -0.01)),
        ('[-0.01, 0)', (values >= -0.01) & (values < 0)),
        ('0', values == 0),
        ('(0, 0.01)', (values > 0) & (values < 0.01)),
        ('[0.01, 0.05)', (values >= 0.01) & (values < 0.05)),
        ('>= 0.05', values >= 0.05),
    ]
    return dict(n=int(values.size), bins=[{'label': label, 'count': int(np.sum(mask))} for label, mask in ordered])
