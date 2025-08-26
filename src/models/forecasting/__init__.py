"""
Forecasting Models Module

This module provides various forecasting models for time series data.
"""

from .DLinear import Model as DLinear
from .PatchTST import Model
PatchTST = Model

__all__ = [
    'DLinear',
    'PatchTST'
]
