# Copyright (c) 2021 Kemal Kurniawan

from typing import Callable

from rnnr.callbacks import save
from sacred.run import Run
import torch


def update_params(opt: torch.optim.Optimizer) -> Callable[[dict], None]:
    def callback(state):
        opt.zero_grad()
        state["loss"].backward()
        opt.step()

    return callback


def log_grads(run: Run, model: torch.nn.Module, every: int = 10) -> Callable[[dict], None]:
    def callback(state):
        if state["n_iters"] % every != 0:
            return

        for name, p in model.named_parameters():
            if p.requires_grad:
                run.log_scalar(f"grad_{name}", p.grad.norm().item(), state["n_iters"])

    return callback


def log_stats(run: Run, every: int = 10) -> Callable[[dict], None]:
    def callback(state):
        if state["n_iters"] % every != 0:
            return

        for name, value in state["stats"].items():
            run.log_scalar(f"batch_{name}", value, state["n_iters"])
        for name, value in state.get("extra_stats", {}).items():
            run.log_scalar(f"batch_{name}", value, state["n_iters"])

    return callback


def save_state_dict(*args, **kwargs) -> Callable[[dict], None]:
    kwargs.update({"using": lambda m, p: torch.save(m.state_dict(), p), "ext": "pth"})
    return save(*args, **kwargs)
