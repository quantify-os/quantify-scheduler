#!/usr/bin/env python
# pylint: disable=wrong-import-position, unused-import, invalid-name
#
# Quantify documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#
import os
import sys

package_path = os.path.abspath("..")
sys.path.insert(0, package_path)

# -- General configuration ---------------------------------------------
# pylint: disable=invalid-name

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",  # auto document docstrings
    "sphinx.ext.napoleon",  # autodoc understands numpy docstrings
    # load after napoleon, improved compatibility with type hints annotations
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx-jsonschema",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx-jsonschema",
    "jupyter_sphinx",
    "sphinx_togglebutton",
    # fancy type hints in docs and
    # solves the same issue as "sphinx_automodapi.smart_resolver"
    # however the smart_resolver seems to fail for external packages like `zhinst`
    "scanpydoc.elegant_typehints",
    "sphinxcontrib.bibtex",
    "quantify_core.sphinx_extensions.notebook_to_jupyter_sphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "qcodes": ("https://qcodes.github.io/Qcodes/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py/", None),
    "dateutil": ("https://dateutil.readthedocs.io/en/stable/", None),
    "fastjsonschema": ("https://horejsek.github.io/python-fastjsonschema/", None),
    "quantify-core": (
        "https://quantify-quantify-core.readthedocs-hosted.com/en/latest/",
        None,
    ),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "qblox-instruments": (
        "https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/",
        None,
    ),
    "zhinst-toolkit": ("https://docs.zhinst.com/zhinst-toolkit/en/latest/", None),
    "zhinst-qcodes": ("https://docs.zhinst.com/zhinst-qcodes/en/v0.1/", None),
}

bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"


# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "Quantify-Scheduler"
copyright = "2020-2022, Qblox & Orange Quantum Systems"
author = "The Quantify consortium"


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Document __init__ docstring together with class doctring (when __init__ is present)
napoleon_include_init_with_doc = True
# NB the line below could be used for a similar result
# BUT the line below ALWAYS includes the __init__ docstring even if it come from the
# parent class which is undesired for analysis subclasses, for example.
# autoclass_content = "both"


# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"


# The name of an image file (relative to this directory) to place at the top
# of the sidebar.

# the white text fits better with the current sphinx theme
# both files are in the repository
# html_logo = "images/QUANTIFY-LOGO.svg"
html_logo = "images/QUANTIFY-LOGO-WHITE-TEXT.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "images/QUANTIFY-FAVICON_16.png"


# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "quantify_scheduler.css",
]


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "quantifydoc"

html_context = {
    "display_gitlab": True,
    "gitlab_user": "quantify-os",
    "gitlab_repo": "quantify-scheduler",
    "gitlab_version": "main/docs/",
}

# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "quantify_scheduler.tex",
        "quantify Documentation",
        "Quantify Consortium ",
        "manual",
    ),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "quantify", "quantify Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "Quantify-Scheduler",
        "Quantify-Scheduler Documentation",
        author,
        "Quantify-Scheduler",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Other Options -----------------------------------------------------

# avoid duplicate label warning even when manual label has been used
suppress_warnings = ["autosectionlabel.*"]

# avoid ugly very long module_a.module_b.module_c.module_d.module_e.module_d.MyClass
# display in docs (very ugly when overflowing the html page width)
# NB the side bar and the link of these objects already includes the full path
add_module_names = False

# Used by scanpydoc.elegant_typehints to correctly link to references to python objects
# that have a mismatch between the python modules real location vs how they are imported
# and documented. These overrides are necessary to fix "reference target not found" when
# these classes are used as type annotations.
# NB Use this only for external packages. Do not do this in quantify and cause problems
# to internal and external developers.
qualname_overrides = {
    # "<true path to module>" : "<API path>"
    "matplotlib.axes._axes.Axes": "matplotlib.axes.Axes",
    "zhinst.qcodes.uhfqa.UHFQA": "zhinst.qcodes.UHFQA",
    "zhinst.qcodes.hdawg.HDAWG": "zhinst.qcodes.HDAWG",
}

numfig = True

autodoc_default_options = {
    "member-order": "groupwise",
    # Ignore any __all__ that might be added accidentally by inexperienced developers
    # This is done to avoid nasty complications with sphinx and its extensions and
    # plenty of "reference target not found" warnings.
    # See also qualname_overrides above, which has to be used for external packages.
    "ignore-module-all": True,
}

# For debugging the CI just add `or True` on the line below
if os.environ.get("GITLAB_CI", "false") == "true":
    print("\n[INFO] Building docs with private-members... See `conf.py` for details.\n")
    # for local build and CI force documentation to build for private members
    # this make sure the docstrings of private members are also correctly formatted, etc
    autodoc_default_options["private-members"] = True

# -- Options for auto documenting typehints ----------------------------

# Please see https://gitlab.com/quantify-os/quantify/-/issues/10 regarding

# below should be imported all "problematic" modules that might raise strange issues
# when building the docs
# e.g., to "partially initialized module", or "most likely due to a circular import"

# This is a practical solution. We tried fixing certain things upstream, e.g.:
# https://github.com/QCoDeS/Qcodes/pull/2909
# but the issues popped up again, so this is the best and easier solution so far

# qcodes imports scipy under the hood but since scipy=1.7.0 it needs to be imported
# here with typing.TYPE_CHECKING = True otherwise we run into quantify-core#
import lmfit  # related to quantify-core#218 and quantify-core#221
import marshmallow
import qcodes

# `pydantic` fails to import automatically and leads to broken documentation,
# if not preloaded.
import pydantic

# When building the docs we need `typing.TYPE_CHECKING` to be `True` so that the
# sphinx' kernel loads the modules corresponding to the typehints and is able to
# auto document types. The modules listed above create issues when loaded with
# `typing.TYPE_CHECKING = True` so we import them beforehand to avoid nasty issues.

# It is a good practice to make use of the following construct to import modules that
# are used for type hints ONLY! the construct is the following:

# if typing.TYPE_CHECKING:
#     import my_expensive_to_import_module as my_module

# NB if you run into any circular import issue it is because you are importing module
# member directly from a module, i.e.:

# if typing.TYPE_CHECKING:
#     from my_expensive_to_import_module import BlaClass # Potential circular import

set_type_checking_flag = True  # this will run `typing.TYPE_CHECKING = True`

notebook_to_jupyter_sphinx_always_rebuild = True

# Enable nitpicky mode - warns about all references where the target cannot be found
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpicky

nitpicky = True  # equivalent to `-n` option in the docs Makefile
nitpick_ignore = []  # Tuple[str, str], ignore certain warnings

with open("nitpick-exceptions.txt", encoding="utf-8") as nitpick_exceptions:
    for line in nitpick_exceptions:
        if line.strip() == "" or line.startswith("#"):
            continue
        dtype, target = line.split(None, 1)
        target = target.strip()
        nitpick_ignore.append((dtype, target))
