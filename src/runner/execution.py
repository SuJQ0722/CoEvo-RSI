"""Shared execution and progress reporting for domain evaluation runtimes."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import islice
from typing import TypeVar

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def in_completion_order(
    items: Iterable[_Input],
    worker: Callable[[_Input], _Output],
    concurrency: int,
) -> Iterator[_Output]:
    """Run independent items and yield each result as soon as it completes."""

    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if concurrency == 1:
        for item in items:
            yield worker(item)
        return
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        iterator = iter(items)
        pending = {
            executor.submit(worker, item)
            for item in islice(iterator, concurrency)
        }
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                pending.add(executor.submit(worker, item))


@dataclass
class ItemProgress:
    """Emit one stable, detailed progress record for every evaluated item."""

    logger: logging.Logger
    total: int
    completed: int = 0

    def log(
        self,
        *,
        sample_id: str,
        score: float,
        **details: object,
    ) -> None:
        self.completed += 1
        fields = [
            f"item={self.completed}/{self.total}",
            f"sample_id={sample_id}",
            f"score={float(score):.6f}",
        ]
        fields.extend(f"{key}={value}" for key, value in details.items())
        self.logger.info("%s", " ".join(fields))
