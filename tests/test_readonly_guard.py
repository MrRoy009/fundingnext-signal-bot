"""Hard constraint 1: the codebase must never be able to place, change or close orders.

The forbidden names are assembled at runtime so this file doesn't trip its own scan.
"""
import ast
from pathlib import Path

import pytest

from data.mt5_adapter import ALLOWED_FUNCTIONS, ReadOnlyMT5
from tests.fakes import FakeMT5

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = {
    "order" + "_send", "order" + "_check",
    "position" + "_close", "positions" + "_close", "close" + "_position",
    "order" + "_modify", "order" + "_delete", "trade" + "_request",
    "TRADE_ACTION" + "_DEAL", "TRADE_ACTION" + "_PENDING", "TRADE_ACTION" + "_SLTP",
    "TRADE_ACTION" + "_MODIFY", "TRADE_ACTION" + "_REMOVE", "TRADE_ACTION" + "_CLOSE_BY",
}
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache"}


def python_files():
    for path in ROOT.rglob("*.py"):
        if not SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            yield path


def test_no_forbidden_names_in_source_text():
    hits = []
    for path in python_files():
        text = path.read_text(encoding="utf-8").lower()
        hits += [f"{path.relative_to(ROOT)}: {name}" for name in FORBIDDEN if name.lower() in text]
    assert not hits, "Order functions found in code:\n" + "\n".join(hits)


def test_no_forbidden_names_in_ast():
    hits = []
    for path in python_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = (node.attr if isinstance(node, ast.Attribute)
                    else node.id if isinstance(node, ast.Name)
                    else node.value if isinstance(node, ast.Constant) and isinstance(node.value, str)
                    else None)
            if name in FORBIDDEN:
                hits.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not hits, "\n".join(hits)


def test_allowlist_has_no_trading_functions():
    assert not ALLOWED_FUNCTIONS & FORBIDDEN


def test_proxy_blocks_order_functions():
    fake = FakeMT5()
    name = "order" + "_send"
    setattr(fake, name, lambda request: pytest.fail("order function reached"))
    proxy = ReadOnlyMT5(fake)
    with pytest.raises(PermissionError):
        getattr(proxy, name)
    with pytest.raises(PermissionError):
        proxy.history_deals_get
    assert proxy.ACCOUNT_TRADE_MODE_DEMO == 0
    assert proxy.symbol_info("USTEC").name == "USTEC"
