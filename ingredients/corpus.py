# Copyright (c) 2021 Kemal Kurniawan

from pathlib import Path

from sacred import Ingredient
from transformers import AutoConfig, AutoTokenizer

from readers import SemEvalTimexReader

ing = Ingredient("corpus")


@ing.config
def default():
    # path to timex data directory
    path = "timex"
    # discard train sentences longer than this
    max_length = 30
    # text encoding
    encoding = "utf8"


@ing.capture
def read_samples(_log, path="timex", max_length=None, encoding="utf8"):
    if max_length is None:
        max_length = 10 ** 9

    _log.info("Reading timex data from %s", path)
    samples = list(SemEvalTimexReader(Path(path), encoding).read_samples())

    _log.info("Tokenizing sentences")
    config = AutoConfig.from_pretrained("clulab/roberta-timex-semeval")
    tokenizer = AutoTokenizer.from_pretrained(
        "clulab/roberta-timex-semeval", config=config, use_fast=True
    )
    sents = [s["sent"] for s in samples]
    tokz = tokenizer(sents, truncation=True, return_offsets_mapping=True)

    for i, s in enumerate(samples):
        s.update({
            "word_ids": tokz["input_ids"][i],
            "mask": tokz["attention_mask"][i],
            "spans": tokz["offset_mapping"][i],
        })
        n_toks = 0
        for j, (beg, end) in enumerate(s["spans"]):
            if beg == end:  # special tokens (e.g., <s>)
                continue
            n_toks += 1
            s["spans"][j] = (beg + s["offset"], end + s["offset"])
        if n_toks > max_length:
            continue
        yield s
