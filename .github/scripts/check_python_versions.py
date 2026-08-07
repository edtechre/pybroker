#!/usr/bin/env python3
"""Assert every Python-version declaration agrees with the CI matrix.

``.github/python-versions.json`` is the single source of truth. The
workflows consume it directly with ``fromJSON()``; the files checked here
cannot, so they are kept in step by hand -- which is exactly how the asv
gate came to benchmark on an interpreter its own config did not list.
This check turns that class of drift into a CI failure.

Run locally from the repository root:

    python .github/scripts/check_python_versions.py

Exits 0 when every site agrees, 1 with one line per disagreement
otherwise.
"""

import configparser
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SSOT = ROOT / ".github" / "python-versions.json"


def toxenv(version: str) -> str:
    """Derives a tox environment name (``3.11`` -> ``py311``)."""
    return "py" + version.replace(".", "")


def version_key(version: str) -> tuple[int, ...]:
    """Sorts version strings numerically, so 3.9 precedes 3.11."""
    return tuple(int(part) for part in version.split("."))


def check_asv_conf(
    versions: list[str], tooling: str, fail: list[str]
) -> None:
    """asv builds one environment per entry in ``pythons``."""
    path = ROOT / "asv.conf.json"
    pythons = json.loads(path.read_text())["pythons"]
    if pythons != versions:
        fail.append(
            f"asv.conf.json: 'pythons' is {pythons}, expected {versions}. "
            "The benchmark suite must declare every supported version so "
            "the PR gate can select each one with --python."
        )


def check_setup_cfg(
    versions: list[str], tooling: str, fail: list[str]
) -> None:
    """setup.cfg carries the tox envlist, the floor, and two pins."""
    cfg = configparser.RawConfigParser()
    cfg.read(ROOT / "setup.cfg")
    floor = min(versions, key=version_key)

    envlist = [
        env.strip()
        for env in cfg.get("tox:tox", "envlist").split(",")
        if env.strip()
    ]
    expected_envlist = [toxenv(version) for version in versions]
    if envlist != expected_envlist:
        fail.append(
            f"setup.cfg [tox:tox]: 'envlist' is {envlist}, expected "
            f"{expected_envlist}."
        )

    python_requires = cfg.get("options", "python_requires").strip()
    if python_requires != f">={floor}":
        fail.append(
            f"setup.cfg [options]: 'python_requires' is "
            f"{python_requires!r}, expected '>={floor}' -- the lowest "
            "version in the matrix."
        )

    mypy_version = cfg.get("mypy", "python_version").strip()
    if mypy_version != floor:
        fail.append(
            f"setup.cfg [mypy]: 'python_version' is {mypy_version!r}, "
            f"expected {floor!r} -- mypy must check against the oldest "
            "supported version, not the newest."
        )

    basepython = cfg.get("testenv:typecheck", "basepython").strip()
    if basepython != f"python{tooling}":
        fail.append(
            f"setup.cfg [testenv:typecheck]: 'basepython' is "
            f"{basepython!r}, expected 'python{tooling}'."
        )


def check_readthedocs(
    versions: list[str], tooling: str, fail: list[str]
) -> None:
    """Read the Docs pins its build interpreter under ``build.tools``.

    Matched with a regex rather than parsed: PyYAML is not a dependency
    of this check, and the pin is a single unambiguous line.
    """
    path = ROOT / ".readthedocs.yml"
    matches = re.findall(
        r'^\s+python:\s*"([^"]+)"\s*$', path.read_text(), re.MULTILINE
    )
    if len(matches) != 1:
        fail.append(
            f".readthedocs.yml: expected exactly one 'python: \"X.Y\"' "
            f"line, found {len(matches)}."
        )
    elif matches[0] != tooling:
        fail.append(
            f".readthedocs.yml: build.tools.python is {matches[0]!r}, "
            f"expected {tooling!r}."
        )


def check_pyproject(
    versions: list[str], tooling: str, fail: list[str]
) -> None:
    """ruff's ``target-version`` selects the syntax level it assumes."""
    path = ROOT / "pyproject.toml"
    with path.open("rb") as handle:
        target = tomllib.load(handle)["tool"]["ruff"]["target-version"]
    supported = [toxenv(version) for version in versions]
    if target not in supported:
        fail.append(
            f"pyproject.toml [tool.ruff]: 'target-version' is {target!r}, "
            f"which is not one of the supported versions {supported}."
        )


def main() -> int:
    ssot = json.loads(SSOT.read_text())
    versions: list[str] = ssot["versions"]
    tooling: str = ssot["tooling"]

    fail: list[str] = []
    if not versions:
        fail.append(f"{SSOT.name}: 'versions' is empty.")
    if versions != sorted(versions, key=version_key):
        fail.append(
            f"{SSOT.name}: 'versions' must be listed oldest first, so the "
            "lowest entry is unambiguously the support floor."
        )
    if tooling not in versions:
        fail.append(
            f"{SSOT.name}: 'tooling' is {tooling!r}, which is not in "
            f"'versions' {versions}. CI would run format, lint, typecheck "
            "and docs on an untested interpreter."
        )

    if not fail:
        check_asv_conf(versions, tooling, fail)
        check_setup_cfg(versions, tooling, fail)
        check_readthedocs(versions, tooling, fail)
        check_pyproject(versions, tooling, fail)

    if fail:
        print(
            f"Python version declarations disagree with {SSOT.name} "
            f"(versions={versions}, tooling={tooling}):\n",
            file=sys.stderr,
        )
        for message in fail:
            print(f"  - {message}", file=sys.stderr)
        print(
            "\nUpdate the site above, or the source of truth if the "
            "matrix itself changed.",
            file=sys.stderr,
        )
        return 1

    print(
        f"All Python version declarations agree with {SSOT.name}: "
        f"versions={versions}, tooling={tooling}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
