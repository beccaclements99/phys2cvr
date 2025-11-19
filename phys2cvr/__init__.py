"""Hopefully importing everything."""

from . import (
    _version,
    io,
    regressors,
    signal,
    stats,
    utils,
    viz,
    workflows,
)

__all__ = [
    io,
    regressors,
    signal,
    stats,
    utils,
    viz,
    workflows,
]

__version__ = _version.get_versions()['version']
