"""Callback types shared across GeoTessera operations."""

from collections.abc import Callable

type ProgressCallback = Callable[[int, int, str], None]
