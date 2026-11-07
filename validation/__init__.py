"""GATE 3 validation tooling (COMMON 10.5, 10.7).

stats:  stationary block bootstrap, regime-conditional trade bootstrap, shuffle test, Deflated
        Sharpe Ratio with N_raw / N_eff, PBO by CSCV. Pure functions on arrays.
replay: competition replay over 30-calendar-day windows (COMMON 10.5.9).

Nothing in this package reads market data or touches the holdout; callers pass arrays built by
the runner from pre-registered configurations only.
"""
