"""asv benchmark suite for pybroker.

Run locally with::

    asv run HEAD^!                  # benchmark current commit
    asv continuous master HEAD      # diff master vs current branch
    asv publish && asv preview      # HTML dashboard

CI runs ``asv continuous origin/master HEAD`` on every PR and posts a
sticky comment (``.github/workflows/asv-pr.yml``).
"""
