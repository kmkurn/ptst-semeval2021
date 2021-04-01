#!/usr/bin/env python

# Copyright (c) 2021 Kemal Kurniawan


from collections import defaultdict
from pathlib import Path
from statistics import median
import math
import os
import pickle
import tempfile

from anafora import AnaforaData
from rnnr import Event, Runner
from rnnr.attachments import EpochTimer, MeanReducer, ProgressBar
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from text2array import BucketIterator, ShuffleIterator
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification
import numpy as np
import torch

from aatrn import compute_ambiguous_tag_pairs_mask
from callbacks import (
    log_grads,
    log_stats,
    save_state_dict,
    update_params,
)
from crf import LinearCRF
from evaluation import score_time
from ingredients.corpus import ing as corpus_ing, read_samples
from models import RoBERTagger
from utils import make_anafora, print_accs

ex = Experiment("sest10-ptst-testrun", ingredients=[corpus_ing])
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Setup mongodb observer
mongo_url = os.getenv("SACRED_MONGO_URL")
db_name = os.getenv("SACRED_DB_NAME")
if None not in (mongo_url, db_name):
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


@ex.config
def default():
    # directory to save finetuning artifacts
    artifacts_dir = "timex_artifacts"
    # whether to overwrite existing artifacts directory
    overwrite = False
    # temperature to regulate confidence (>1 means less confident)
    temperature = 1.0
    # whether to freeze the embedding layers
    freeze_embeddings = True
    # freeze encoder earlier layers up to this layer
    freeze_encoder_up_to = 5
    # device to run on [cpu, cuda]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # cumulative prob threshold
    thresh = 0.95
    # batch size
    batch_size = 50
    # learning rate
    lr = 1e-5
    # max number of epochs
    max_epoch = 5
    # path to directory containing the gold annotations
    gold_path = ""
    # whether to write predictions when finetuning is finished
    predict_on_finished = False
    # load model parameters from here (evaluate)
    load_params = "model.pth"
    # whether to save confusion matrix (evaluate)
    save_confusion_matrix = False


@ex.named_config
def best():
    lr = 9e-6
    temperature = 2.56


@ex.capture
def run_eval(
    model,
    id2label,
    samples,
    corpus,
    _log,
    device="cpu",
    batch_size=32,
    gold_path="",
    compute_loss=False,
    confusion=False,
):
    if not gold_path and not compute_loss:
        _log.info("Skipping evaluation since gold data isn't provided and loss isn't required")
        return None, None

    runner = Runner()
    runner.state.update({"preds": [], "_ids": []})

    @runner.on(Event.BATCH)
    def maybe_compute_prediction(state):
        if not gold_path:
            return

        arr = state["batch"].to_array()
        state["arr"] = arr
        assert arr["mask"].all()
        words = torch.from_numpy(arr["word_ids"]).long().to(device)

        model.eval()
        scores = model(words)
        preds = LinearCRF(scores).argmax()

        state["preds"].extend(preds.tolist())
        state["_ids"].extend(arr["_id"].tolist())
        if compute_loss:
            state["scores"] = scores

    @runner.on(Event.BATCH)
    def maybe_compute_loss(state):
        if not compute_loss:
            return

        arr = state["arr"] if "arr" in state else state["batch"].to_array()
        state["arr"] = arr
        if "scores" in state:
            scores = state["scores"]
        else:
            assert arr["mask"].all()
            words = torch.from_numpy(arr["word_ids"]).long().to(device)
            model.eval()
            scores = model(words)

        mask = torch.from_numpy(arr["mask"]).bool().to(device)
        ptst_mask = torch.from_numpy(arr["ptst_mask"]).bool().to(device)

        masked_scores = scores.masked_fill(~ptst_mask, -1e9)
        crf = LinearCRF(masked_scores)
        crf_z = LinearCRF(scores)
        ptst_loss = -crf.log_partitions().sum() + crf_z.log_partitions().sum()
        state["ptst_loss"] = ptst_loss.item()
        state["size"] = mask.size(0)

    @runner.on(Event.BATCH)
    def set_n_items(state):
        state["n_items"] = int(state["arr"]["mask"].sum())

    n_tokens = sum(len(s["word_ids"]) for s in samples)
    ProgressBar(leave=False, total=n_tokens, unit="tok").attach_on(runner)
    if compute_loss:
        MeanReducer("mean_ptst_loss", value="ptst_loss").attach_on(runner)

    with torch.no_grad():
        runner.run(BucketIterator(samples, lambda s: len(s["word_ids"]), batch_size))

    if runner.state["preds"]:
        assert len(runner.state["preds"]) == len(samples)
        assert len(runner.state["_ids"]) == len(samples)
        for i, preds in zip(runner.state["_ids"], runner.state["preds"]):
            samples[i]["preds"] = preds

    if gold_path:
        group = defaultdict(list)
        for s in samples:
            group[str(s["path"])].append(s)

        with tempfile.TemporaryDirectory() as dirname:
            dirname = Path(dirname)
            for doc_path, doc_samples in group.items():
                spans = [x for s in doc_samples for x in s["spans"]]
                labels = [id2label[x] for s in doc_samples for x in s["preds"]]
                doc_path = Path(doc_path[len(f"{corpus['path']}/") :])
                data = make_anafora(spans, labels, doc_path.name)
                (dirname / doc_path.parent).mkdir(parents=True, exist_ok=True)
                data.to_file(f"{str(dirname / doc_path)}.xml")
            return (
                score_time(gold_path, str(dirname), confusion),
                runner.state.get("mean_ptst_loss"),
            )
    return None, runner.state.get("mean_ptst_loss")


@ex.capture
def read_samples_(_log, **kwargs):
    samples = list(read_samples(**kwargs))
    for i, s in enumerate(samples):
        s["_id"] = i
    n_toks = sum(len(s["word_ids"]) for s in samples)
    _log.info("Read %d samples and %d tokens", len(samples), n_toks)
    return samples


@ex.command(unobserved=True)
def evaluate_src_model(_log, _run, device="cpu"):
    """Evaluate the source model."""

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            emissions = self.model(*args, **kwargs)[0]
            bsz, slen, nl = emissions.shape

            scores = emissions[:, :-1].unsqueeze(2)
            assert scores.shape == (bsz, slen - 1, 1, nl)
            scores = scores.expand(bsz, slen - 1, nl, nl)

            scores = scores.clone()
            scores[:, -1] += emissions[:, -1].unsqueeze(2)
            assert scores.shape == (bsz, slen - 1, nl, nl)

            return scores

    model_name = "clulab/roberta-timex-semeval"
    _log.info("Loading %s", model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = Wrapper(AutoModelForTokenClassification.from_pretrained(model_name, config=config))
    model.to(device)

    _log.info("Evaluating")
    eval_score, _ = run_eval(model, config.id2label, read_samples_())
    print_accs(eval_score)

    return eval_score["f1"]


@ex.command
def evaluate(
    _log,
    _run,
    temperature=1.0,
    artifacts_dir="artifacts",
    load_params="model.pth",
    device="cpu",
    save_confusion_matrix=False,
):
    """Evaluate a trained target model."""
    model_name = "clulab/roberta-timex-semeval"
    _log.info("Loading %s", model_name)
    config = AutoConfig.from_pretrained(model_name)
    token_clf = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    model = RoBERTagger(token_clf, config.num_labels, temperature)

    artifacts_dir = Path(artifacts_dir)
    _log.info("Loading model parameters from %s", artifacts_dir / load_params)
    model.load_state_dict(torch.load(artifacts_dir / load_params, "cpu"))
    model.to(device)

    _log.info("Evaluating")
    eval_score, _ = run_eval(model, config.id2label, read_samples_(), confusion=save_confusion_matrix)
    c = eval_score.pop("confusion", None)
    print_accs(eval_score, on="test", run=_run)
    if c is not None:
        labels = set()
        for k in c.keys():
            labels.update(k)
        if "O" in labels:
            labels.remove("O")
        labels = sorted(labels)
        labels.insert(0, "O")

        label2id = {l: i for i, l in enumerate(labels)}
        m = np.zeros((len(labels), len(labels)))
        for k, cnt in c.items():
            m[label2id[k[0]], label2id[k[1]]] = cnt

        _log.info("Saving labels list in %s", artifacts_dir / "labels.pkl")
        with open(artifacts_dir / "labels.pkl", "wb") as f:
            pickle.dump(labels, f)
        _log.info("Saving confusion matrix in %s", artifacts_dir / "confusion.npy")
        np.save(artifacts_dir / "confusion.npy", m)

    return eval_score["f1"]


@ex.command(unobserved=True)
def report_coverage(
    corpus, _log, temperature=1.0, device="cpu", batch_size=16, thresh=0.95, gold_path=""
):
    """Report coverage of gold tags in the chart."""
    samples = read_samples_()
    model_name = "clulab/roberta-timex-semeval"
    _log.info("Loading %s", model_name)
    config = AutoConfig.from_pretrained(model_name)
    token_clf = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    model = RoBERTagger(token_clf, config.num_labels, temperature)

    _log.info("Initializing transitions")
    torch.nn.init.zeros_(model.start_transition)
    torch.nn.init.zeros_(model.transition)
    for lid, label in config.id2label.items():
        if not label.startswith("I-"):
            continue

        with torch.no_grad():
            model.start_transition[lid] = -1e9
        for plid, plabel in config.id2label.items():
            if plabel == "O" or plabel[2:] != label[2:]:
                with torch.no_grad():
                    model.transition[plid, lid] = -1e9

    model.to(device)

    _log.info("Computing ambiguous PTST tag pairs mask")
    model.eval()
    ptst_masks, _ids = [], []
    pbar = tqdm(total=sum(len(s["word_ids"]) for s in samples), unit="tok")
    for batch in BucketIterator(samples, lambda s: len(s["word_ids"]), batch_size):
        arr = batch.to_array()
        assert arr["mask"].all()
        words = torch.from_numpy(arr["word_ids"]).long().to(device)
        with torch.no_grad():
            ptst_mask = compute_ambiguous_tag_pairs_mask(model(words), thresh)
        ptst_masks.extend(ptst_mask.tolist())
        _ids.extend(arr["_id"].tolist())
        pbar.update(int(arr["mask"].sum()))
    pbar.close()

    assert len(ptst_masks) == len(samples)
    assert len(_ids) == len(samples)
    for i, ptst_mask in zip(_ids, ptst_masks):
        samples[i]["ptst_mask"] = ptst_mask

    _log.info("Reporting coverage of gold labels")

    group = defaultdict(list)
    for s in samples:
        k = str(s["path"])[len(f"{corpus['path']}/") :]
        group[k].append(s)

    n_cov_tp, n_total_tp, n_cov_ts, n_total_ts = 0, 0, 0, 0
    for dirpath, _, filenames in os.walk(gold_path):
        if not filenames:
            continue
        if len(filenames) > 1:
            raise ValueError(f"more than 1 file is found in {dirpath}")
        if not filenames[0].endswith(".TimeNorm.gold.completed.xml"):
            raise ValueError(f"{filenames[0]} doesn't have the expected suffix")

        doc_path = os.path.join(dirpath, filenames[0])
        data = AnaforaData.from_file(doc_path)
        prefix, suffix = f"{gold_path}/", ".TimeNorm.gold.completed.xml"
        doc_path = doc_path[len(prefix) : -len(suffix)]
        tok_spans = [p for s in group[doc_path] for p in s["spans"]]
        tok_spans.sort()

        labeling = {}
        for ann in data.annotations:
            if len(ann.spans) != 1:
                raise ValueError("found annotation with >1 span")
            span = ann.spans[0]
            beg = 0
            while beg < len(tok_spans) and tok_spans[beg][0] < span[0]:
                beg += 1
            end = beg
            while end < len(tok_spans) and tok_spans[end][1] < span[1]:
                end += 1
            if (
                beg < len(tok_spans)
                and end < len(tok_spans)
                and tok_spans[beg][0] == span[0]
                and tok_spans[end][1] == span[1]
                and beg not in labeling
            ):
                labeling[beg] = f"B-{ann.type}"
                for i in range(beg + 1, end + 1):
                    if i not in labeling:
                        labeling[i] = f"I-{ann.type}"

        labels = ["O"] * len(tok_spans)
        for k, v in labeling.items():
            labels[k] = v

        offset = 0
        for s in group[doc_path]:
            ts_covd = True
            for i in range(1, len(s["spans"])):
                plab = labels[offset + i - 1]
                lab = labels[offset + i]
                if s["ptst_mask"][i - 1][config.label2id[plab]][config.label2id[lab]]:
                    n_cov_tp += 1
                else:
                    ts_covd = False
                n_total_tp += 1
            if ts_covd:
                n_cov_ts += 1
            n_total_ts += 1
            offset += len(s["spans"])

    _log.info(
        "Number of covered tag pairs: %d out of %d (%.1f%%)",
        n_cov_tp,
        n_total_tp,
        100.0 * n_cov_tp / n_total_tp,
    )
    _log.info(
        "Number of covered tag sequences: %d out of %d (%.1f%%)",
        n_cov_ts,
        n_total_ts,
        100.0 * n_cov_ts / n_total_ts,
    )


@ex.automain
def finetune(
    _log,
    _run,
    _rnd,
    corpus,
    artifacts_dir="artifacts",
    overwrite=False,
    temperature=1.0,
    freeze_embeddings=True,
    freeze_encoder_up_to=1,
    device="cpu",
    thresh=0.95,
    batch_size=16,
    lr=1e-5,
    max_epoch=5,
    predict_on_finished=False,
):
    """Finetune/train the source model on unlabeled target data."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(exist_ok=overwrite)

    samples = read_samples_()
    eval_samples = read_samples_(max_length=None)
    model_name = "clulab/roberta-timex-semeval"
    _log.info("Loading %s", model_name)
    config = AutoConfig.from_pretrained(model_name)
    token_clf = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    model = RoBERTagger(token_clf, config.num_labels, temperature)

    _log.info("Initializing transitions")
    torch.nn.init.zeros_(model.start_transition)
    torch.nn.init.zeros_(model.transition)
    for lid, label in config.id2label.items():
        if not label.startswith("I-"):
            continue

        with torch.no_grad():
            model.start_transition[lid] = -1e9
        for plid, plabel in config.id2label.items():
            if plabel == "O" or plabel[2:] != label[2:]:
                with torch.no_grad():
                    model.transition[plid, lid] = -1e9

    for name, p in model.named_parameters():
        freeze = False
        if freeze_embeddings and ".embeddings." in name:
            freeze = True
        if freeze_encoder_up_to >= 0:
            for i in range(freeze_encoder_up_to + 1):
                if f".encoder.layer.{i}." in name:
                    freeze = True
        if freeze:
            _log.info("Freezing %s", name)
            p.requires_grad_(False)

    model.to(device)

    _log.info("Computing ambiguous PTST tag pairs mask")
    model.eval()
    ptst_masks, _ids = [], []
    pbar = tqdm(total=sum(len(s["word_ids"]) for s in samples), unit="tok")
    for batch in BucketIterator(samples, lambda s: len(s["word_ids"]), batch_size):
        arr = batch.to_array()
        assert arr["mask"].all()
        words = torch.from_numpy(arr["word_ids"]).long().to(device)
        with torch.no_grad():
            ptst_mask = compute_ambiguous_tag_pairs_mask(model(words), thresh)
        ptst_masks.extend(ptst_mask.tolist())
        _ids.extend(arr["_id"].tolist())
        pbar.update(int(arr["mask"].sum()))
    pbar.close()

    assert len(ptst_masks) == len(samples)
    assert len(_ids) == len(samples)
    for i, ptst_mask in zip(_ids, ptst_masks):
        samples[i]["ptst_mask"] = ptst_mask

    _log.info("Report number of sequences")
    log_total_nseqs, log_nseqs = [], []
    pbar = tqdm(total=sum(len(s["word_ids"]) for s in samples), leave=False)
    for batch in BucketIterator(samples, lambda s: len(s["word_ids"]), batch_size):
        arr = batch.to_array()
        assert arr["mask"].all()
        ptst_mask = torch.from_numpy(arr["ptst_mask"]).bool().to(device)
        cnt_scores = torch.zeros_like(ptst_mask).float()
        cnt_scores_masked = cnt_scores.masked_fill(~ptst_mask, -1e9)
        log_total_nseqs.extend(LinearCRF(cnt_scores).log_partitions().tolist())
        log_nseqs.extend(LinearCRF(cnt_scores_masked).log_partitions().tolist())
        pbar.update(arr["word_ids"].size)
    pbar.close()
    cov = [math.exp(x - x_) for x, x_ in zip(log_nseqs, log_total_nseqs)]
    _log.info(
        "Number of seqs: min {:.2} ({:.2}%) | med {:.2} ({:.2}%) | max {:.2} ({:.2}%)".format(
            math.exp(min(log_nseqs)),
            100 * min(cov),
            math.exp(median(log_nseqs)),
            100 * median(cov),
            math.exp(max(log_nseqs)),
            100 * max(cov),
        )
    )

    _log.info("Creating optimizer")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    finetuner = Runner()

    @finetuner.on(Event.BATCH)
    def compute_loss(state):
        arr = state["batch"].to_array()
        words = torch.from_numpy(arr["word_ids"]).long().to(device)
        mask = torch.from_numpy(arr["mask"]).bool().to(device)
        ptst_mask = torch.from_numpy(arr["ptst_mask"]).bool().to(device)

        model.train()
        scores = model(words, mask)
        masked_scores = scores.masked_fill(~ptst_mask, -1e9)

        # mask passed to LinearCRF shouldn't include the last token
        last_idx = mask.long().sum(dim=1, keepdim=True) - 1
        mask_ = mask.scatter(1, last_idx, False)[:, :-1]

        crf = LinearCRF(masked_scores, mask_)
        crf_z = LinearCRF(scores, mask_)
        ptst_loss = -crf.log_partitions().sum() + crf_z.log_partitions().sum()
        ptst_loss /= mask.size(0)

        state["loss"] = ptst_loss
        state["stats"] = {"ptst_loss": ptst_loss.item()}
        state["n_items"] = mask.long().sum().item()

    finetuner.on(Event.BATCH, [update_params(opt), log_grads(_run, model), log_stats(_run)])

    @finetuner.on(Event.EPOCH_FINISHED)
    def evaluate(state):
        _log.info("Evaluating on train")
        eval_score, loss = run_eval(model, config.id2label, samples, compute_loss=True)
        if eval_score is not None:
            print_accs(eval_score, on="train", run=_run, step=state["n_iters"])
        _log.info("train_ptst_loss: %.4f", loss)
        _run.log_scalar("train_ptst_loss", loss, step=state["n_iters"])

        _log.info("Evaluating on eval")
        eval_score, _ = run_eval(model, config.id2label, eval_samples)
        if eval_score is not None:
            print_accs(eval_score, on="eval", run=_run, step=state["n_iters"])

        state["eval_f1"] = None if eval_score is None else eval_score["f1"]

    finetuner.on(Event.EPOCH_FINISHED, save_state_dict("model", model, under=artifacts_dir))

    @finetuner.on(Event.FINISHED)
    def maybe_predict(state):
        if not predict_on_finished:
            return

        _log.info("Computing predictions")
        model.eval()
        preds, _ids = [], []
        pbar = tqdm(total=sum(len(s["word_ids"]) for s in eval_samples), unit="tok")
        for batch in BucketIterator(eval_samples, lambda s: len(s["word_ids"]), batch_size):
            arr = batch.to_array()
            assert arr["mask"].all()
            words = torch.from_numpy(arr["word_ids"]).long().to(device)
            scores = model(words)
            pred = LinearCRF(scores).argmax()
            preds.extend(pred.tolist())
            _ids.extend(arr["_id"].tolist())
            pbar.update(int(arr["mask"].sum()))
        pbar.close()

        assert len(preds) == len(eval_samples)
        assert len(_ids) == len(eval_samples)
        for i, preds_ in zip(_ids, preds):
            eval_samples[i]["preds"] = preds_

        group = defaultdict(list)
        for s in eval_samples:
            group[str(s["path"])].append(s)

        _log.info("Writing predictions")
        for doc_path, doc_samples in group.items():
            spans = [x for s in doc_samples for x in s["spans"]]
            labels = [config.id2label[x] for s in doc_samples for x in s["preds"]]
            doc_path = Path(doc_path[len(f"{corpus['path']}/") :])
            data = make_anafora(spans, labels, doc_path.name)
            (artifacts_dir / "time" / doc_path.parent).mkdir(parents=True, exist_ok=True)
            data.to_file(
                f"{str(artifacts_dir / 'time' / doc_path)}.TimeNorm.system.completed.xml"
            )

    EpochTimer().attach_on(finetuner)
    n_tokens = sum(len(s["word_ids"]) for s in samples)
    ProgressBar(stats="stats", total=n_tokens, unit="tok").attach_on(finetuner)

    bucket_key = lambda s: (len(s["word_ids"]) - 1) // 10
    trn_iter = ShuffleIterator(
        BucketIterator(samples, bucket_key, batch_size, shuffle_bucket=True, rng=_rnd),
        rng=_rnd,
    )
    _log.info("Starting finetuning")
    try:
        finetuner.run(trn_iter, max_epoch)
    except KeyboardInterrupt:
        _log.info("Interrupt detected, training will abort")
    else:
        return finetuner.state.get("eval_f1")
