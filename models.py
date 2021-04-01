# Copyright (c) 2021 Kemal Kurniawan

from transformers import RobertaForTokenClassification as RobertaClf
import torch
import torch.nn as nn


class RoBERTagger(nn.Module):
    def __init__(self, token_clf: RobertaClf, n_labels: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.token_clf = token_clf
        self.temperature = temperature
        self.start_transition = nn.Parameter(torch.empty(n_labels))  # type: ignore
        self.transition = nn.Parameter(torch.empty(n_labels, n_labels))  # type: ignore
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transition)
        nn.init.uniform_(self.transition)

    def forward(self, words, mask=None):
        bsz, slen = words.shape
        nl = self.start_transition.size(0)

        emissions = self.token_clf(words, mask)[0]
        assert emissions.shape == (bsz, slen, nl)
        emissions /= self.temperature
        scores = emissions[:, :-1].unsqueeze(2) + self.transition.t()
        assert scores.shape == (bsz, slen - 1, nl, nl)
        scores[:, 0] += self.start_transition
        scores[:, -1] += emissions[:, -1].unsqueeze(2)
        return scores
