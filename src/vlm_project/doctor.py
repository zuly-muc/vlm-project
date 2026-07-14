"""Environment/health checks for the ``vlm-project doctor`` command.

Verifies that the runtime has everything the default (local BLIP) path needs:
core imports, a working SQLite FTS5, the heavy ML stack, the nuScenes devkit, and
whether BLIP weights are already cached (baked into the image) vs. will download
on first use. Critical failures exit non-zero; a missing weight cache is a warning.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass

_BLIP_DEFAULT = "Salesforce/blip-image-captioning-base"


@dataclass
class HealthCheck:
    name: str
    ok: bool
    critical: bool
    detail: str = ""


def _try_import(module: str) -> tuple[bool, str]:
    try:
        mod = __import__(module)
        return True, getattr(mod, "__version__", "ok")
    except Exception as e:  # noqa: BLE001 - report any import failure verbatim
        return False, str(e)


def _fts5_available() -> tuple[bool, str]:
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        con.close()
        return True, "fts5 ok"
    except sqlite3.OperationalError as e:
        return False, str(e)


def _weights_cached(model_name: str) -> tuple[bool, str]:
    try:
        from transformers import BlipProcessor

        BlipProcessor.from_pretrained(model_name, local_files_only=True)
        return True, "weights cached locally"
    except Exception as e:  # noqa: BLE001
        return False, f"not cached ({type(e).__name__}); will download at runtime"


def run_checks(model_name: str = _BLIP_DEFAULT) -> list[HealthCheck]:
    checks: list[HealthCheck] = []

    ok = sys.version_info >= (3, 10)
    checks.append(HealthCheck("python>=3.10", ok, True,
                              ".".join(map(str, sys.version_info[:3]))))

    for mod, critical in [("numpy", True), ("PIL", True), ("torch", True),
                          ("transformers", True), ("nuscenes", False)]:
        ok, detail = _try_import(mod)
        checks.append(HealthCheck(f"import {mod}", ok, critical, detail))

    ok, detail = _fts5_available()
    checks.append(HealthCheck("sqlite fts5", ok, True, detail))

    ok, detail = _weights_cached(model_name)
    checks.append(HealthCheck("blip weights cached", ok, False, detail))

    return checks


def render(checks: list[HealthCheck]) -> str:
    lines = []
    for c in checks:
        if c.ok:
            mark = "PASS"
        else:
            mark = "FAIL" if c.critical else "WARN"
        lines.append(f"  [{mark}] {c.name}" + (f": {c.detail}" if c.detail else ""))
    return "\n".join(lines)


def main(model_name: str = _BLIP_DEFAULT) -> int:
    checks = run_checks(model_name)
    print(render(checks))
    critical_ok = all(c.ok for c in checks if c.critical)
    print("  => " + ("HEALTHY" if critical_ok else "UNHEALTHY (critical checks failed)"))
    return 0 if critical_ok else 1
