"""Tiny vector helper used to avoid mandatory NumPy for core logic."""
from __future__ import annotations


class FloatVector(list[float]):
    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            return FloatVector(result)
        return result

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self),)

    def tolist(self) -> list[float]:
        return list(self)
