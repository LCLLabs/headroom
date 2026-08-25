"""Reducer package exports."""

from headroom.proxy.explore_pruner.reducers.coact import COACT_NAME, CoactReducer
from headroom.proxy.explore_pruner.reducers.swe_pruner import SWE_PRUNER_NAME, SwePrunerReducer

__all__ = ["COACT_NAME", "CoactReducer", "SWE_PRUNER_NAME", "SwePrunerReducer"]
