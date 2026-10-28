"""Exercise real optimizer steps with local and standalone remote trace collectors."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.training_trace import TraceCollector


@pytest.fixture(params=["local", "remote"])
def collector_factory(request, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    if request.param == "local":
        return TraceCollector
    script = Path(__file__).resolve().parents[2] / "agent/monitor_agent.py"
    spec = importlib.util.spec_from_file_location("sampling_agent", script)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    namespace = {"__name__": "trace_sampling_test"}
    exec(compile(agent.REMOTE_TRACE_RUNNER, "<remote-trace>", "exec"), namespace)
    remote = namespace["Collector"]
    remote.on_train_start = remote.start
    remote.on_train_batch_start = remote.batch_start
    remote.on_train_batch_end = remote.batch_end
    remote.on_train_end = remote.end
    return remote


def trainer_with_optimizer():
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.ReLU())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return SimpleNamespace(model=model, optimizer=optimizer, preprocess_batch=lambda value: value,
                           epoch=0, loss_items={})


@pytest.mark.parametrize("failure_at", [1, 7], ids=["pre-step", "post-step"])
def test_sampling_failure_in_optimizer_does_not_interrupt_training(
    collector_factory, tmp_path, monkeypatch, failure_at,
):
    import torch

    trainer = trainer_with_optimizer()
    collector = collector_factory(tmp_path / "trace", maximum=2, interval=2)
    original = trainer.preprocess_batch
    collector.on_train_start(trainer)
    collector.on_train_batch_start(trainer)
    for parameter in trainer.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    previous = [parameter.detach().clone() for parameter in trainer.model.parameters()]

    cpu, calls = torch.Tensor.cpu, []

    def unavailable_statistics(tensor, *args, **kwargs):
        calls.append(True)
        if len(calls) == failure_at:
            raise RuntimeError("simulated diagnostic tensor transfer failure")
        return cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", unavailable_statistics)
    # Real torch hook iteration also checks that exception recovery does not
    # mutate the optimizer hook dictionaries while they are being iterated.
    trainer.optimizer.step()
    collector.on_train_batch_end(trainer)
    assert collector.disabled
    for parameter, before in zip(trainer.model.parameters(), previous):
        assert torch.allclose(parameter, before - 0.1)
    assert not trainer.optimizer._optimizer_step_pre_hooks
    assert not trainer.optimizer._optimizer_step_post_hooks
    assert trainer.preprocess_batch is original
    assert not any(module._forward_hooks for module in trainer.model.modules())


def test_interval_skips_all_tensor_statistics_and_limit_releases_hooks(
    collector_factory, tmp_path, monkeypatch,
):
    import torch

    trainer = trainer_with_optimizer()
    collector = collector_factory(tmp_path / "trace", maximum=2, interval=2)
    original = trainer.preprocess_batch
    collector.on_train_start(trainer)
    cpu_calls = []
    cpu = torch.Tensor.cpu

    def counted_cpu(tensor, *args, **kwargs):
        cpu_calls.append(True)
        return cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    statistics_per_batch = []
    for _ in range(6):
        collector.on_train_batch_start(trainer)
        trainer.preprocess_batch({})
        trainer.optimizer.zero_grad()
        trainer.model(torch.ones(2, 2)).sum().backward()
        trainer.optimizer.step()
        collector.on_train_batch_end(trainer)
        statistics_per_batch.append(len(cpu_calls))
    assert statistics_per_batch[0] > 0
    assert statistics_per_batch[1] == statistics_per_batch[0]
    assert statistics_per_batch[2] > statistics_per_batch[1]
    assert statistics_per_batch[3:] == [statistics_per_batch[2]] * 3
    assert [sample["batch"] for sample in collector.samples] == [0, 2]
    assert trainer.preprocess_batch is original
    assert not trainer.optimizer._optimizer_step_pre_hooks
    assert not trainer.optimizer._optimizer_step_post_hooks
    assert not any(module._forward_hooks for module in trainer.model.modules())
    assert collector.last_batch == {}
    assert (tmp_path / "trace/trace.json").is_file()


def test_no_sampling_budget_keeps_training_unwrapped(collector_factory, tmp_path):
    trainer = trainer_with_optimizer()
    original = trainer.preprocess_batch
    collector = collector_factory(tmp_path / "trace", maximum=0, interval=1)
    collector.on_train_start(trainer)
    assert trainer.preprocess_batch is original
    assert not trainer.optimizer._optimizer_step_pre_hooks
    assert not any(module._forward_hooks for module in trainer.model.modules())


@pytest.mark.parametrize("labeled", [False, True])
def test_tensor_loss_items_from_real_trainer_are_sampled_without_boolean_conversion(
    collector_factory, tmp_path, labeled,
):
    import torch

    trainer = trainer_with_optimizer()
    trainer.loss_items = torch.tensor([1.0, 2.0, 3.0])
    if labeled:
        trainer.label_loss_items = lambda values: dict(zip(["box_loss", "cls_loss", "dfl_loss"], values))
    collector = collector_factory(tmp_path / "trace", maximum=1, interval=1)
    collector.on_train_start(trainer)
    collector.on_train_batch_start(trainer)
    trainer.preprocess_batch({})
    collector.on_train_batch_end(trainer)
    assert not collector.disabled
    expected = {"box_loss": 1.0, "cls_loss": 2.0, "dfl_loss": 3.0} if labeled else {
        "loss_0": 1.0, "loss_1": 2.0, "loss_2": 3.0}
    assert collector.samples[0]["loss"] == expected
