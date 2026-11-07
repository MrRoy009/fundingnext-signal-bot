"""Strategies cleared for live tickets.

Only add a strategy here after QA has signed its validation report and the owner has
approved it at GATE 4. Until then the engine issues no tickets.
"""
from __future__ import annotations

from strategies.base import Strategy

APPROVED: list[Strategy] = []
