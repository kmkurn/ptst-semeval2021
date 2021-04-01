# Copyright (c) 2021 Kemal Kurniawan

from typing import Iterator
from pathlib import Path
import os

from spacy.lang.en import English


class SemEvalTimexReader:
    def __init__(self, root_dir: Path, encoding: str = "utf8") -> None:
        self.root_dir = root_dir
        self.enc = encoding
        nlp = English()
        nlp.add_pipe(nlp.create_pipe("sentencizer"))
        self._nlp = nlp

    def read_samples(self) -> Iterator[dict]:
        for dirpath, _, filenames in os.walk(self.root_dir):
            for name in filenames:
                with open(Path(dirpath) / name, encoding=self.enc) as f:
                    text = f.read()
                doc = self._nlp(text)
                offset = 0
                for sent in doc.sents:
                    sent = sent.text_with_ws
                    yield {"sent": sent, "path": Path(dirpath) / name, "offset": offset}
                    offset += len(sent)
