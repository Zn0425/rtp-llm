"""DSV4 Torch profiler ranges."""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from typing import Any, Callable, ContextManager, Optional, Type

import torch

_RANGES_ENABLED = os.environ.get("DSV4_RECORD_FUNCTION_RANGES", "1") != "0"
_DISABLED_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "dsv4_record_function_ranges_disabled", default=0
)


class _NoopRecordFunctionRange:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb,
    ) -> bool:
        return False


_NOOP_RECORD_FUNCTION_RANGE = _NoopRecordFunctionRange()

LayerForwardRange = Callable[[int], ContextManager[Any]]


def record_function_ranges_enabled() -> bool:
    return _RANGES_ENABLED and _DISABLED_DEPTH.get() <= 0


def record_function_range(name: str):
    if not record_function_ranges_enabled():
        return _NOOP_RECORD_FUNCTION_RANGE
    return torch.profiler.record_function(name)


def _noop_layer_forward_range(_layer_idx: int) -> ContextManager[Any]:
    return _NOOP_RECORD_FUNCTION_RANGE


def _active_layer_forward_range(layer_idx: int) -> ContextManager[Any]:
    # FUNCTION scope retains CPU layer attribution without projecting a user
    # annotation onto every GPU stream covered by the range.
    return torch._C._profiler._RecordFunctionFast(f"forward(layer={layer_idx})")


def make_layer_forward_range() -> LayerForwardRange:
    """Capture profiler state before the fast path suppresses nested ranges."""
    if not record_function_ranges_enabled():
        return _noop_layer_forward_range
    profiler_enabled = getattr(torch.autograd, "_profiler_enabled", None)
    if profiler_enabled is not None and not profiler_enabled():
        return _noop_layer_forward_range
    return _active_layer_forward_range


@contextmanager
def disable_record_function_ranges():
    depth = _DISABLED_DEPTH.get()
    token = _DISABLED_DEPTH.set(depth + 1)
    try:
        yield
    finally:
        _DISABLED_DEPTH.reset(token)
