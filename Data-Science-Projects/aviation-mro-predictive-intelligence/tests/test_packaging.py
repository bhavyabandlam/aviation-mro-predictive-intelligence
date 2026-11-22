"""Packaging consistency tests (Phase B1).

Guards that ``pyproject.toml`` and ``requirements.txt`` stay in sync and that the
package metadata is coherent. These tests are static (no network, no install) and
run in the default test suite.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
LICENSE = PROJECT_ROOT / "LICENSE"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _load_pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def _parse_requirements() -> dict[str, str]:
    """Return {normalized_name: version_spec} from requirements.txt."""
    pins: dict[str, str] = {}
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*([=<>!~]=?\s*[^#]+)?", line)
        if not match:
            continue
        name = _normalize(match.group(1))
        spec = (match.group(2) or "").strip()
        pins[name] = spec
    return pins


def _core_dependencies() -> list[str]:
    data = _load_pyproject()
    return list(data["project"]["dependencies"])


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return _load_pyproject()


@pytest.fixture(scope="module")
def requirements() -> dict[str, str]:
    return _parse_requirements()


def test_pyproject_parses_and_has_core_metadata(pyproject: dict) -> None:
    project = pyproject["project"]
    assert project["name"] == "aviation-mro"
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.13"
    assert project["readme"] == "README.md"


def test_packages_are_discovered_under_src(pyproject: dict) -> None:
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"]
    assert "aviation_mro*" in find["include"]
    # The package must actually exist with an __init__.
    assert (PROJECT_ROOT / "src" / "aviation_mro" / "__init__.py").is_file()


def test_core_dependencies_are_pinned(pyproject: dict) -> None:
    deps = _core_dependencies()
    names = {_normalize(re.split(r"[=<>!~]", dep)[0]) for dep in deps}
    required = {
        "pandas",
        "numpy",
        "pyarrow",
        "scipy",
        "scikit-learn",
        "joblib",
        "statsmodels",
        "pyyaml",
        "matplotlib",
        "streamlit",
        "shap",
    }
    assert required.issubset(names), f"missing core deps: {required - names}"


def test_requirements_covers_every_core_dependency(
    pyproject: dict, requirements: dict[str, str]
) -> None:
    deps = _core_dependencies()
    for dep in deps:
        name = _normalize(re.split(r"[=<>!~]", dep)[0])
        assert name in requirements, f"{name} is in pyproject but missing from requirements.txt"


def test_requirements_pins_match_pyproject(pyproject: dict, requirements: dict[str, str]) -> None:
    for dep in _core_dependencies():
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([^;]+)$", dep)
        assert match, f"pyproject dependency is not pinned with '==': {dep}"
        name = _normalize(match.group(1))
        want = match.group(2).strip()
        assert requirements.get(name) == f"=={want}", (
            f"version mismatch for {name}: pyproject '=={want}' vs requirements "
            f"'{requirements.get(name)}'"
        )


def test_dev_optional_dependencies_declared(pyproject: dict) -> None:
    extras = pyproject["project"]["optional-dependencies"]
    assert "dev" in extras
    dev_names = {_normalize(re.split(r"[=<>!~]", d)[0]) for d in extras["dev"]}
    assert {"pandas-stubs", "scipy-stubs", "types-pyyaml"}.issubset(dev_names)


def test_license_is_mit() -> None:
    assert LICENSE.is_file(), "LICENSE file is missing"
    text = LICENSE.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Permission is hereby granted, free of charge" in text
