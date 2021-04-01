# Copyright (c) 2021 Kemal Kurniawan

from einops import rearrange
from torch import BoolTensor, Tensor

from crf import LinearCRF


def compute_ambiguous_tag_pairs_mask(scores: Tensor, threshold: float = 0.95) -> BoolTensor:
    bsz, slen, n_next_tags, n_tags = scores.shape

    crf = LinearCRF(scores)
    margs = crf.marginals()

    # select high prob tag pairs until their cumulative probability exceeds threshold
    margs = rearrange(margs, "bsz slen nntags ntags -> bsz slen (nntags ntags)")
    margs, orig_indices = margs.sort(dim=2, descending=True)
    tp_mask = margs.cumsum(dim=2) < threshold

    # select the tag pairs that make the cum sum exceeds threshold
    last_idx = tp_mask.long().sum(dim=2, keepdim=True).clamp(max=n_next_tags * n_tags - 1)
    tp_mask = tp_mask.scatter(2, last_idx, True)

    # restore the order and shape
    _, restore_indices = orig_indices.sort(dim=2)
    tp_mask = tp_mask.gather(2, restore_indices)

    # ensure best tag sequence is selected
    best_tags = crf.argmax()
    assert best_tags.shape == (bsz, slen + 1)
    best_idx = best_tags[:, 1:] * n_tags + best_tags[:, :-1]
    assert best_idx.shape == (bsz, slen)
    tp_mask = tp_mask.scatter(2, best_idx.unsqueeze(2), True)

    tp_mask = rearrange(
        tp_mask, "bsz slen (nntags ntags) -> bsz slen nntags ntags", nntags=n_next_tags
    )
    return tp_mask  # type: ignore
