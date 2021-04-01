# Copyright (c) 2021 Kemal Kurniawan

import collections

from anafora_eval import evaluate as anafora_evaluate


def score_time(ref_domain, res_domain, confusion=False):
    scores_type = anafora_evaluate.Scores
    exclude = "Event"
    file_named_scores = anafora_evaluate.score_dirs(
        reference_dir=ref_domain, predicted_dir=res_domain, exclude=exclude
    )  # pairwise=True

    all_named_scores = collections.defaultdict(lambda: scores_type())
    for _, named_scores in file_named_scores:
        for name, scores in named_scores.items():
            all_named_scores[name].update(scores)

    scores = {
        "f1": all_named_scores["*"].f1(),
        "precision": all_named_scores["*"].precision(),
        "recall": all_named_scores["*"].recall(),
    }
    if confusion:
        scores["confusion"] = all_named_scores["*"].confusion
    return scores
