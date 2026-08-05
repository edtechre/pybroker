import os
import sys

sys.path.insert(0, os.path.abspath("../../src/"))
sys.path.insert(0, os.path.abspath("notebooks/"))

nitpick_ignore = [
    ("py:class", "type"),
    ("py:class", "numpy.float64"),
    ("py:class", "numpy.int64"),
    ("py:class", "numpy._typing._array_like._ScalarT"),
    # numpy < 2.3 spells the above TypeVar differently.
    ("py:class", "numpy._typing._array_like._ScalarType_co"),
    # pandas < 3.0 exposes internal module paths in type annotations.
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "pandas.core.series.Series"),
    ("py:class", "pandas._libs.tslibs.timestamps.Timestamp"),
    # optuna exposes internal module paths in type annotations, and its
    # docs register optuna.Study only under optuna.study.Study.
    ("py:class", "optuna.study.study.Study"),
    ("py:class", "optuna.trial._trial.Trial"),
    ("py:class", "optuna.Study"),
    # Annotations are written with import aliases (np., pd., bare Decimal /
    # datetime / "Stop"), which napoleon copies verbatim into attribute
    # types where they cannot resolve.
    ("py:class", "np.ndarray"),
    ("py:class", "np.datetime64"),
    ("py:class", "np.floating"),
    ("py:class", "pd.DataFrame"),
    ("py:class", "pd.Series"),
    ("py:class", "Decimal"),
    ("py:class", "datetime"),
    ("py:class", "'Stop'"),
    # Private classes referenced by type annotations are not documented.
    ("py:class", "BaseSampler"),
    ("py:class", "pybroker.scope._StoreBacking"),
    ("py:class", "_ExecutionsHost"),
    ("py:class", "_OptimizeTrialHost"),
]

project = "PyBroker"
copyright = "2026, Edward West"
author = "Edward West"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.jquery",
]

autodoc_default_options = {
    "special-members": "__call__",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"

add_module_names = False
autosummary_generate = True
keep_warnings = False
# Included SKILL.md bodies start at ## under an RST skill subtitle.
suppress_warnings = ["myst.header"]

python_version = ".".join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    "python": ("https://docs.python.org/" + python_version, None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "diskcache": ("https://grantjenks.com/docs/diskcache/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "PyBroker"
html_theme = "sphinx_rtd_theme"
html_static_path = ["../_static"]
html_extra_path = ["../_html"]
# Set canonical URL from the Read the Docs Domain
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
html_context = {}
# Tell Jinja2 templates the build is running on Read the Docs
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

# Multi-language docs.
language = "en"
locales = ["zh_CN"]
locale_dirs = ["../locales/"]
gettext_compact = False
gettext_uuid = True

# Framework types/functions stay on the indicator module page, but are omitted
# from the index toctree under Indicators (built-in indicators only).
_INDICATOR_TOC_EXCLUDE = frozenset(
    {
        "pybroker.indicator.Indicator",
        "pybroker.indicator.IndicatorSet",
        "pybroker.indicator.IndicatorsMixin",
        "pybroker.indicator.indicator",
    }
)


def _omit_indicator_framework_from_toc(app, doctree):
    if app.env.docname != "reference/pybroker.indicator":
        return
    from sphinx import addnodes

    for desc in doctree.findall(addnodes.desc):
        for child in desc:
            if not isinstance(child, addnodes.desc_signature):
                continue
            if not _INDICATOR_TOC_EXCLUDE.intersection(child.get("ids", [])):
                continue
            # Hide the object and its members; otherwise methods bubble up
            # to the top-level Indicators toctree on the index.
            for nested in desc.findall(addnodes.desc):
                nested["no-contents-entry"] = True
            break


def _shorten_config_toc_names(app, doctree):
    """Drop the StrategyConfig. prefix in the Configuration Options TOC.

    Mutates ``env.tocs`` after TocTreeCollector so the index sidebar picks
    up the short names.
    """
    if app.env.docname != "reference/pybroker.config":
        return
    from docutils import nodes

    prefix = "StrategyConfig."
    toc = app.env.tocs.get(app.env.docname)
    if toc is None:
        return
    for ref in toc.findall(nodes.reference):
        text = ref.astext()
        if not text.startswith(prefix):
            continue
        short = text[len(prefix) :]
        ref.clear()
        ref.append(nodes.literal("", short))


def setup(app):
    # Run before TocTreeCollector (default priority 500).
    app.connect(
        "doctree-read", _omit_indicator_framework_from_toc, priority=400
    )
    # After TocTreeCollector has built env.tocs for this doc.
    app.connect("doctree-read", _shorten_config_toc_names, priority=600)
