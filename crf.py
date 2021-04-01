# Copyright (c) 2021 Kemal Kurniawan

from typing import Optional

from torch import BoolTensor, LongTensor, Tensor
from torch_struct import LinearChainCRF
import torch


class LinearCRF:
    def __init__(self, scores: Tensor, mask: Optional[BoolTensor] = None) -> None:
        bsz, slen = scores.shape[:2]
        if mask is None:
            mask = scores.new_full([bsz, slen], 1).bool()  # type: ignore
        assert mask is not None
        lengths = mask.type_as(scores).sum(dim=1)
        self._crf = LinearChainCRF(scores, lengths=lengths + 1)
        self._mask = mask

    def log_partitions(self) -> Tensor:
        return self._crf.partition

    def argmax(self) -> LongTensor:
        amax = self._crf.argmax
        bsz, slen, n_next_tags, n_tags = amax.shape
        assert n_next_tags == n_tags

        lengths = self._mask.long().sum(dim=1)
        if (lengths != slen).any():
            raise NotImplementedError("argmax is only implemented for same-length sequences")

        amax, max_next_tags = amax.max(dim=2)
        assert amax.shape == (bsz, slen, n_tags) and max_next_tags.shape == amax.shape

        _, max_tags = amax.max(dim=2)
        assert max_tags.shape == (bsz, slen)

        max_last_tags = max_next_tags[:, -1].gather(1, max_tags[:, -1].unsqueeze(1))
        assert max_last_tags.shape == (bsz, 1)

        return torch.cat([max_tags, max_last_tags], dim=1)  # type: ignore

    def marginals(self) -> Tensor:
        return self._crf.marginals
