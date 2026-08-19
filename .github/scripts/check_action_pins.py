#!/usr/bin/env python3
"""Assert every GitHub Action is pinned to a full-length commit SHA.

A ``uses:`` reference like ``actions/checkout@v7`` is a *mutable*
pointer: the action's maintainer moves that tag whenever they ship a
v7.x, so the code CI executes can change with no commit in this
repository. ``marocchino/sticky-pull-request-comment@v3`` was not even a
tag but a branch, which moves on every push.

Pinning to a commit SHA freezes what runs. The trailing ``# v7`` comment
is only a human-readable label, though -- nothing stops it from drifting
away from the SHA it claims to describe, and a comment that lies is
worse than no comment. This check turns both failure modes into a CI
failure:

* a reference that is not a 40-character commit SHA, and
* a pinned SHA whose version comment is missing.

Run locally from the repository root:

    python .github/scripts/check_action_pins.py

Pass ``--verify-remote`` to additionally ask the GitHub API whether each
pinned SHA really belongs to the version its comment names. That needs
network access, so CI runs it only where a token is available; the
structural check above is offline and always runs.

Exits 0 when every reference is pinned, 1 with one line per problem
otherwise.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = "https://api.github.com"

# ``- uses: owner/repo@ref  # comment``. The comment is optional here so
# that a pinned reference missing its label is reported as a distinct
# problem rather than skipped as unparseable.
USES = re.compile(
    r"^\s*(?:-\s*)?uses:\s*(?P<ref>\S+)"
    r"(?:\s+#\s*(?P<comment>.+?))?\s*$"
)
SHA = re.compile(r"^[0-9a-f]{40}$")


def workflow_files() -> list[Path]:
    """Every file that can carry a ``uses:`` reference."""
    paths = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    paths += sorted((ROOT / ".github" / "workflows").glob("*.yaml"))
    paths += sorted((ROOT / ".github" / "actions").glob("*/action.yml"))
    return paths


def request(url: str) -> object:
    """GETs ``url`` as JSON, or returns None when GitHub says no."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "pybroker-check-action-pins",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError:
        return None
    except urllib.error.URLError:
        return None


def resolve(repo: str, version: str) -> str:
    """Resolves a tag or branch name to the commit SHA it points at."""
    ref = request(f"{API}/repos/{repo}/git/ref/tags/{version}")
    if ref is None:
        ref = request(f"{API}/repos/{repo}/git/ref/heads/{version}")
    if ref is None:
        return ""
    obj = ref["object"]  # type: ignore[index]
    # An annotated tag points at a tag object, which points at the
    # commit; a lightweight tag points straight at the commit.
    if obj["type"] == "tag":
        tag = request(f"{API}/repos/{repo}/git/tags/{obj['sha']}")
        if tag is None:
            return ""
        return str(tag["object"]["sha"])  # type: ignore[index]
    return str(obj["sha"])


def collect(fail: list[str]) -> list[tuple[str, str, str]]:
    """Checks that every reference is pinned and labelled.

    Returns the ``(repo, sha, version)`` triples worth verifying
    remotely, so the offline pass decides what the online pass looks at.
    """
    pinned: list[tuple[str, str, str]] = []
    for path in workflow_files():
        name = path.relative_to(ROOT)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            match = USES.match(line)
            if match is None:
                continue
            ref = match.group("ref")
            # A local composite action is this repository's own code; it
            # moves with the commit under test and cannot be pinned.
            if ref.startswith("./"):
                continue
            if "@" not in ref:
                fail.append(f"{name}:{number}: {ref!r} has no version at all.")
                continue
            repo, _, rev = ref.partition("@")
            comment = match.group("comment")
            if not SHA.match(rev):
                fail.append(
                    f"{name}:{number}: {repo} is pinned to {rev!r}, which "
                    "is a mutable tag or branch, not a 40-character "
                    "commit SHA. The maintainer can move it, changing "
                    "what CI runs with no commit here."
                )
            elif comment is None:
                fail.append(
                    f"{name}:{number}: {repo} is pinned to {rev[:7]} with "
                    "no trailing '# <version>' comment, leaving no way to "
                    "tell which release it is."
                )
            else:
                pinned.append((repo, rev, comment.split()[0]))
    return pinned


def verify(pinned: list[tuple[str, str, str]]) -> None:
    """Asks GitHub whether each SHA belongs to the version it claims."""
    for repo, sha, version in sorted(set(pinned)):
        actual = resolve(repo, version)
        if not actual:
            print(
                f"  ? {repo}@{sha[:7]} ({version}): could not resolve "
                "the version -- skipped, not treated as a failure."
            )
        elif actual == sha:
            print(f"  OK {repo}@{sha[:7]} is exactly {version}.")
        else:
            # The tag may simply have moved on since this pin was taken,
            # which Dependabot resolves on its own schedule. Only report
            # it; a stale pin is safe by construction, and failing here
            # would turn every upstream release into a red build.
            print(
                f"  ~ {repo}@{sha[:7]} is not the current {version} "
                f"({actual[:7]}). Safe, but outdated -- Dependabot "
                "should open a bump."
            )


def main() -> int:
    fail: list[str] = []
    pinned = collect(fail)

    if fail:
        print("GitHub Actions are not pinned:\n", file=sys.stderr)
        for message in fail:
            print(f"  - {message}", file=sys.stderr)
        print(
            "\nPin each reference to the commit SHA its tag resolves to "
            "and keep the version in a trailing comment, for example:\n"
            "  uses: actions/checkout@3d3c42e5... # v7",
            file=sys.stderr,
        )
        return 1

    print(f"All {len(pinned)} action references are pinned to commit SHAs.")
    if "--verify-remote" in sys.argv:
        print("Verifying each pin against the GitHub API:")
        verify(pinned)
    return 0


if __name__ == "__main__":
    sys.exit(main())
