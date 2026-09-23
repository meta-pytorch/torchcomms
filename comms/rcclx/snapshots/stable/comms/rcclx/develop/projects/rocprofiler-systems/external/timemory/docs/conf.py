# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# sys.path.insert(0, os.path.abspath('.'))

import os
import re
import sys
import glob
import shutil
import fileinput
import subprocess as sp

from recommonmark.transform import AutoStructify
from recommonmark.parser import CommonMarkParser
from pygments.styles import get_all_styles
import recommonmark

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))


def install(package):
    sp.call([sys.executable, "-m", "pip", "install", package])


# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get("READTHEDOCS", None) == "True"


# -- Project information -----------------------------------------------------
project = "timemory"
copyright = "2020,  The Regents of the University of California"
author = "Jonathan R. Madsen"

version = open(os.path.join("..", "VERSION")).read().strip()
# The full version, including alpha/beta/rc tags
release = version

_docdir = os.path.realpath(os.getcwd())
_srcdir = os.path.realpath(os.path.join(os.getcwd(), ".."))
_bindir = os.path.realpath(os.path.join(os.getcwd(), "build-timemory"))
_doxbin = os.path.realpath(os.path.join(_bindir, "doc"))
_doxdir = os.path.realpath(
    os.path.join(_docdir, "_build", "html", "doxygen-docs")
)
_xmldir = os.path.realpath(os.path.join(_docdir, "doxygen-xml"))
_sitedir = os.path.realpath(os.path.join(os.getcwd(), "..", "site"))
_staticdir = os.path.realpath(os.path.join(_docdir, "_static"))
_templatedir = os.path.realpath(os.path.join(_docdir, "_templates"))

if not os.path.exists(_staticdir):
    os.makedirs(_staticdir)

if not os.path.exists(_templatedir):
    os.makedirs(_templatedir)


def build_doxy_docs():
    if not os.path.exists(_bindir):
        os.makedirs(_bindir)
    os.chdir(_bindir)
    sp.run(
        [
            "cmake",
            "-DTIMEMORY_BUILD_DOCS=ON",
            "-DENABLE_DOXYGEN_XML_DOCS=ON",
            "-DENABLE_DOXYGEN_HTML_DOCS=ON",
            "-DENABLE_DOXYGEN_LATEX_DOCS=OFF",
            "-DENABLE_DOXYGEN_MAN_DOCS=OFF",
            "-DTIMEMORY_BUILD_KOKKOS_TOOLS=ON",
            "-DTIMEMORY_BUILD_C=OFF",
            "-DTIMEMORY_BUILD_CUDA=OFF",
            "-DTIMEMORY_BUILD_FORTRAN=OFF",
            _srcdir,
        ]
    )
    sp.run(["cmake", "--build", os.getcwd(), "--target", "doc"])


def prep_doxy_docs():
    """Removes any previously copied generated documentation from build folder"""
    if os.path.exists(_doxdir):
        shutil.rmtree(_doxdir)
    if os.path.exists(_xmldir):
        shutil.rmtree(_xmldir)


def clean_doxy_docs():
    """Removes the generated doxygen documentation"""
    os.chdir(_docdir)
    if os.path.exists(_bindir):
        shutil.rmtree(_bindir)


def copy_doxy_docs():
    """Copies the generated doxygen documentation and README files"""
    os.chdir(_docdir)
    shutil.copytree(os.path.join(_doxbin, "html"), _doxdir)
    shutil.copytree(os.path.join(_doxbin, "xml"), _xmldir)
    shutil.copyfile(
        os.path.join(_bindir, "doc", "Doxyfile.timemory"),
        os.path.join(_docdir, "Doxyfile.timemory"),
    )
    for t in [
        "timemory-timem",
        "timemory-run",
        "timemory-mpip",
        "timemory-ompt",
        "timemory-ncclp",
        "timemory-mallocp",
        "timemory-avail",
        "timemory-jump",
        "timemory-stubs",
        "timemory-nvml",
        "kokkos-connector",
        "timemory-compiler-instrument",
    ]:
        shutil.copyfile(
            os.path.join(_srcdir, "source", "tools", t, "README.md"),
            os.path.join(_docdir, "tools", t, "README.md"),
        )

    shutil.copyfile(
        os.path.join(_srcdir, "source", "python", "README.md"),
        os.path.join(_docdir, "api", "python.md"),
    )

    shutil.copyfile(
        os.path.join(_srcdir, "CONTRIBUTING.md"),
        os.path.join(_docdir, "CONTRIBUTING.md"),
    )


# set to avoid rebuilding doxygen every time
develop = os.environ.get("TIMEMORY_DOCS_DEV", None)

prep_doxy_docs()

if develop is None:
    build_doxy_docs()

copy_doxy_docs()

if develop is None:
    clean_doxy_docs()

# remove known issues
if False:
    os.chdir(_xmldir)
    for file in glob.glob("*.xml"):
        print(f"cleaning macros from {file}...")
        for line in fileinput.input(file, inplace=True):
            for key in [
                "TIMEMORY_NEVER_INSTRUMENT" "TIMEMORY_EXTERN_VISIBLE",
                "TIMEMORY_ALWAYS_INLINE",
                "TIMEMORY_NODISCARD",
                "TIMEMORY_NOINLINE" "TIMEMORY_FLATTEN",
                "TIMEMORY_HIDDEN",
                "TIMEMORY_NOCLONE",
                "TIMEMORY_INLINE" "TIMEMORY_HOT",
                "TIMEMORY_LAMBDA",
                "TIMEMORY_DEVICE_LAMBDA",
                "TIMEMORY_DEVICE_FUCTION",
                "TIMEMORY_GLOBAL_FUNCTION",
                "TIMEMORY_HOST_DEVICE_FUNCTION",
                "TIMEMORY_DEVICE_INLINE",
                "TIMEMORY_GLOBAL_INLINE",
                "TIMEMORY_HOST_DEVICE_INLINE",
                " TIMEMORY_VISIBLE",
                "TIMEMORY_DLL ",
                "TIMEMORY_CDLL ",
            ]:
                if key in line:
                    line = line.replace(key, "")
            if "TIMEMORY_VISIBILITY" in line:
                line = re.sub(r"TIMEMORY_VISIBILITY\((\S+)\)", "", line)
            print(line, end="")
    os.chdir(_docdir)


# -- General configuration ---------------------------------------------------

install("sphinx_rtd_theme")

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_markdown_tables",
    "recommonmark",
    "breathe",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

source_parsers = {".md": CommonMarkParser}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

default_role = None

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "analytics_id": "G-R0682G8FP1",
    "analytics_anonymize_ip": False,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "vcs_pageview_mode": "",
    # 'style_nav_header_background': 'white',
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Breathe Configuration
breathe_projects = {"timemory": "doxygen-xml"}
breathe_default_project = "timemory"
# breathe_default_members = ('members', )
if False:
    breathe_projects_source = {
        "auto": (
            "../source",
            [
                "timemory/compat/library.h",
                "timemory/manager.hpp",
                "timemory/storage/declaration.hpp",
                "python/libpytimemory-api.cpp",
                "python/libpytimemory-auto-timer.cpp",
                "python/libpytimemory-component-bundle.cpp",
                "python/libpytimemory-component-list.cpp",
                "python/libpytimemory-components.cpp",
                "python/libpytimemory-components.hpp",
                "python/libpytimemory-enumeration.cpp",
                "python/libpytimemory-hardware-counters.cpp",
                "python/libpytimemory-rss-usage.cpp",
                "python/libpytimemory-settings.cpp",
                "python/libpytimemory-signals.cpp",
                "python/libpytimemory-units.cpp",
                "python/libpytimemory.cpp",
                "python/libpytimemory.hpp",
            ],
        )
    }

# The name of the Pygments (syntax highlighting) style to use.
styles = list(get_all_styles())
preferences = ("emacs", "pastie", "colorful")
for pref in preferences:
    if pref in styles:
        pygments_style = pref
        break


# app setup hook
def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {
            "auto_toc_tree_section": "Contents",
            "enable_eval_rst": True,
            "enable_auto_doc_ref": False,
        },
        True,
    )
    app.add_transform(AutoStructify)
