# Copyright (c) 2021 Kemal Kurniawan

from typing import Mapping, Optional, Sequence, Tuple
from warnings import warn
import logging

from anafora import AnaforaData, AnaforaEntity
from sacred.run import Run


logger = logging.getLogger(__name__)


def print_accs(
    accs: Mapping[str, float],
    on: str = "dev",
    run: Optional[Run] = None,
    step: Optional[int] = None,
) -> None:
    for key, acc in accs.items():
        logger.info(f"{on}_{key}: {acc:.2%}")
        if run is not None:
            run.log_scalar(f"{on}_{key}", acc, step=step)


def make_anafora(
    spans: Sequence[Tuple[int, int]], labels: Sequence[str], doc_name: str = "DOC_NAME"
) -> AnaforaData:
    data = AnaforaData()

    def add_annotation(start, end, label):
        ent = AnaforaEntity()
        n_ents = len(data.xml.findall("annotations/entity"))
        ent.id = f"{n_ents}@{doc_name}"
        ent.spans = ((start, end),)
        ent.type = label
        data.annotations.append(ent)

    pending = None
    for i in range(len(spans)):
        label, (span_beg, span_end) = labels[i], spans[i]
        if label.startswith("B-"):
            if pending is not None:
                add_annotation(*pending)
            pending = (span_beg, span_end, label[2:])
        elif label.startswith("I-"):
            if pending is not None and label[2:] == pending[2]:
                pending = (pending[0], span_end, pending[2])
            else:
                warn(f"{label} isn't preceded by the begin label, will be treated as O")
                if pending is not None:
                    add_annotation(*pending)
                    pending = None
        elif label == "O" and pending is not None:
            add_annotation(*pending)
            pending = None
        elif label != "O":
            raise ValueError(f"unknown label {label}")
    if pending is not None:
        add_annotation(*pending)

    return data
