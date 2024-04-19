#!/usr/bin/env python
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
import re
import sys
from typing import Any, Dict

package_path = os.path.abspath("..")
sys.path.insert(0, package_path)


# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",  # auto document docstrings
    "sphinx.ext.napoleon",  # autodoc understands numpy docstrings
    # load after napoleon, improved compatibility with type hints annotations
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx-jsonschema",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "jupyter_sphinx",
    "sphinx_togglebutton",
    # fancy type hints in docs and
    # solves the same issue as "sphinx_automodapi.smart_resolver"
    # however the smart_resolver seems to fail for external packages like `zhinst`
    "scanpydoc.elegant_typehints",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
    "autoapi.extension",
    "sphinx_design",
]


# required to use sphinx_design in combination with myst
myst_enable_extensions = ["colon_fence"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

autoapi_template_dir = "_templates"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "qcodes": ("https://microsoft.github.io/Qcodes/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py/", None),
    "dateutil": ("https://dateutil.readthedocs.io/en/stable/", None),
    "fastjsonschema": ("https://horejsek.github.io/python-fastjsonschema/", None),
    "quantify-core": (
        "https://quantify-os.org/docs/quantify-core/dev/",
        None,
    ),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "qblox-instruments": (
        "https://qblox-qblox-instruments.readthedocs-hosted.com/en/main/",
        None,
    ),
    # FIXME change the "*objects.inv" strings in the lines below back to None, and
    # remove the local files, once https://github.com/zhinst/zhinst-toolkit/issues/278
    # is resolved.
    "zhinst-toolkit": (
        "https://docs.zhinst.com/zhinst-toolkit/en/latest/",
        "zhinst_toolkit_objects.inv",
    ),
    "zhinst-qcodes": (
        "https://docs.zhinst.com/zhinst-qcodes/en/v0.1/",
        "zhinst_qcodes_objects.inv",
    ),
}

bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "label"


# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "quantify-scheduler"
copyright = "2020-2024, Qblox & Orange Quantum Systems"
author = "Quantify Consortium"


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "dev/profiling/metrics.py",
    "dev/profiling/random_gates.py",
    "dev/profiling/resonator_spectroscopy.py",
    "dev/profiling/simple_binned_acquisition.py",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "images/QUANTIFY-FAVICON_16.png"


# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options: Dict[str, Any] = {
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "logo": {
        "image_light": "QUANTIFY_SCHEDULER_FLAT.svg",
        "image_dark": "QUANTIFY_SCHEDULER_FLAT_DM.svg",
    },
    "navigation_with_keys": False,
    "show_version_warning_banner": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
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
        "Quantify Documentation",
        "Quantify Consortium",
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
        "quantify-scheduler",
        "Quantify-scheduler Documentation",
        author,
        "quantify-scheduler",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Other Options -----------------------------------------------------
# see https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

autoapi_type = "python"
autoapi_generate_api_docs = True
autoapi_dirs = ["../../quantify_scheduler"]
ignore_module_all = True
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    # Including `important-members` displays the description of class aliases in the
    # docs, however, it causes sphinx to raise multiple warnings about finding
    # multiple targets for cross-references.
    # "imported-members",
]
# displays docstrings inside __init__
autoapi_python_class_content = "class"

# avoid duplicate label warning even when manual label has been used;
suppress_warnings = [
    "autosectionlabel.*",
    "mystnb.unknown_mime_type",
    "mystnb.mime_priority",
]

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

# qcodes0.36.0 lazyloads h5py which causes build failures
import h5py

# qcodes imports scipy under the hood but since scipy=1.7.0 it needs to be imported
# here with typing.TYPE_CHECKING = True otherwise we run into quantify-core#
import lmfit  # related to quantify-core#218 and quantify-core#221
import marshmallow

# `pydantic` fails to import automatically and leads to broken documentation,
# if not preloaded.
import pydantic
import qcodes

# Prevents a circular import warning
import tenacity

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

# Automatically generate anchors for MyST headers
myst_heading_anchors = 3

# By default execution mode is set to "cache": that allowes to store execution result
# to local cache. However, for a purpose of faster editing we allow to override it
# locally using MYSTNB_EXECUTION_MODE environment variable. That is useful if
# documentation author uses `sphinx-autobuild`: setting it to "off" will disable
# automated execution completely, and documentation author will be able to edit
# documentation in JupyterLab or VSCode and see live changes in a browser with
# minimal delay possible.
nb_execution_mode = os.environ.get("MYSTNB_EXECUTION_MODE") or "cache"

# The following fails the build when one of the notebooks has an execution error,
# and this error is not allowed explicitly.
nb_execution_raise_on_error = True

# Default cell execution timeout.
nb_execution_timeout = 120

# Exclude notebooks from execution
nb_execution_excludepatterns = ["source/dev/profiling/*.ipynb"]

# Configure pydata-sphinx-theme version switcher based on detected CI environment
# variables.
if (git_tag := os.environ.get("CI_COMMIT_TAG")) is not None and re.match(
    r"^v([0-9]+)\.([0-9]+)\.([0-9]+)((rc|a|b)([0-9]+))?$", git_tag
):
    switcher_version = git_tag
elif (
    (branch := os.environ.get("CI_COMMIT_BRANCH"))
    and (default_branch := os.environ.get("CI_DEFAULT_BRANCH"))
    and branch == default_branch
):
    switcher_version = "dev"
else:
    switcher_version = None

if switcher_version is not None:
    html_theme_options["switcher"] = {
        "json_url": "https://quantify-os.org/docs/quantify-scheduler/switcher.json",
        "version_match": switcher_version,
    }
    html_theme_options["navbar_center"] = ["version-switcher", "navbar-nav"]

# The following fails the build when one of the notebooks has an execution error.
nb_execution_raise_on_error = True

# Making sure sphinx builders understand cells which include multiple mime types,
# for example linkcheck can check cells which contain html and images
nb_mime_priority_overrides = [
    ("linkcheck", "text/html", 0),
    ("linkcheck", "image/png", 40),
]

# Workaround for sphinxcontrib.mermaid bug:
# mermaid.min.js needs to be loaded after require.min.js
# see https://github.com/mgaitan/sphinxcontrib-mermaid/issues/124
mermaid_js_priority = 100

# These are working links but (the redirect) doesn't allow polling
linkcheck_ignore = [
    "https://doi.org/10.1063/1.447644",
    "https://doi.org/10.1063/1.5089550",
    "https://doi.org/10.1063/1.5133894",
    "https://doi.org/10.1109/TQE.2020.2965810",
    "https://doi.org/10.4233/uuid:78155c28-3204-4130-a645-a47e89c46bc5",
    "https://www.sciencedirect.com/science/article/pii/S0370157313000562",
    "dot",  # To not require graphviz in linkcheck image
]

# Enable nitpicky mode - warns about all references where the target cannot be found
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpicky

nitpicky = True  # equivalent to `-n` option in the docs Makefile
nitpick_ignore = [
    ("py:class", "quantify_scheduler.device_under_test.composite_square_edge"),
    (
        "py:class",
        "quantify_scheduler.schedules.schedule.CompiledSchedule.hardware_waveform_dict",
    ),
    (
        "py:class",
        "quantify_scheduler.schedules.schedule.CompiledSchedule.hardware_timing_table",
    ),
    ("py:class", "DictOrdered"),
    ("py:class", "quantify_scheduler.helpers.schedule.AcquisitionMetadata"),
    ("py:class", "_StaircaseParameters"),
    ("py:class", "quantify_scheduler.operations.operation.Operation"),
    ("py:class", "quantify_scheduler.operations.operation.Operation.valid_pulse"),
    (
        "py:class",
        "quantify_scheduler.instrument_coordinator.instrument_coordinator.InstrumentCoordinator",
    ),
    (
        "py:class",
        "quantify_scheduler.instrument_coordinator.components.InstrumentCoordinatorComponentBase",
    ),
    ("py:class", "AcquisitionIndexing"),
    ("py:class", "optional"),
    ("py:obj", "quantify_scheduler.Operation"),
    ("py:class", "quantify_scheduler.Operation"),
    ("py:obj", "quantify_scheduler.Resource"),
    ("py:class", "quantify_scheduler.Resource"),
    ("py:obj", "quantify_scheduler.structure.DataStructure"),
    ("py:obj", "quantify_scheduler.backends.SerialCompiler"),
    ("py:obj", "quantify_scheduler.backends.qblox.operations.StitchedPulseBuilder"),
    ("py:obj", "quantify_scheduler.backends.qblox.operations.long_ramp_pulse"),
    ("py:obj", "quantify_scheduler.backends.qblox.operations.long_square_pulse"),
    ("py:obj", "quantify_scheduler.backends.qblox.operations.staircase_pulse"),
    ("py:obj", "quantify_scheduler.schedules.heterodyne_spec_sched"),
    ("py:obj", "quantify_scheduler.schedules.heterodyne_spec_sched_nco"),
    ("py:obj", "quantify_scheduler.schedules.nv_dark_esr_sched"),
    ("py:obj", "quantify_scheduler.schedules.two_tone_spec_sched"),
    ("py:obj", "quantify_scheduler.schedules.two_tone_spec_sched_nco"),
    ("py:obj", "quantify_scheduler.schedules.allxy_sched"),
    ("py:obj", "quantify_scheduler.schedules.echo_sched"),
    ("py:obj", "quantify_scheduler.schedules.rabi_pulse_sched"),
    ("py:obj", "quantify_scheduler.schedules.rabi_sched"),
    ("py:obj", "quantify_scheduler.schedules.ramsey_sched"),
    ("py:obj", "quantify_scheduler.schedules.readout_calibration_sched"),
    ("py:obj", "quantify_scheduler.schedules.t1_sched"),
    ("py:obj", "quantify_scheduler.schedules.trace_schedule"),
    ("py:obj", "quantify_scheduler.schedules.trace_schedule_circuit_layer"),
    ("py:obj", "quantify_scheduler.schedules.two_tone_trace_schedule"),
    ("py:class", "SUPPORTED_ACQ_PROTOCOLS"),
    ("py:class", "HardwareDescription"),
    ("py:class", "LatencyCorrection"),
    ("py:class", "quantify_scheduler.backends.types.qblox.QbloxHardwareDescription"),
    ("py:class", "quantify_scheduler.backends.types.zhinst.ZIHardwareDescription"),
    ("py:class", "ClusterModuleDescription"),
    ("py:class", "Literal[Zurich Instruments]"),
    ("py:class", "Literal[Mock readout module]"),
    ("py:class", "RealInputGain"),
    ("py:class", "OutputAttenuation"),
    ("py:class", "InputAttenuation"),
    ("py:class", "typing.AnnotatedAlias"),
    ("py:obj", "quantify_scheduler.structure.NDArray"),
    ("py:obj", "quantify_scheduler.structure.Graph"),
    ("py:attr", "BasicTransmonElement.measure.acq_threshold"),
    ("py:attr", "BasicTransmonElement.measure.acq_rotation"),
]  # Tuple[str, str], ignore certain warnings

nitpick_ignore_regex = [
    ("py:class", r"numpy.*"),
    ("py:class", r"np.*"),
    ("py:.*", r"pydantic.*"),
    ("py:.*", r"qcodes.*"),
    ("py:class", r"Ellipsis.*"),
    ("py:class", r"Parameter.*"),
    ("py:.*", r"dataclasses_json.*"),
    (".*", r".*Schedule"),
    ("py:class", r"qblox_instruments.*"),
    ("py:class", r"matplotlib.*"),
    ("py:class", r"mpl.*"),
    ("py:class", r"\"[a-zA-Z]+\""),  # Ignore string literals
    ("py:class", r".*\.Self"),
    ("py:class", r"dataclasses.*"),
]

with open("nitpick-exceptions.txt", encoding="utf-8") as nitpick_exceptions:
    for line in nitpick_exceptions:
        if line.strip() == "" or line.startswith("#"):
            continue
        dtype, target = line.split(None, 1)
        target = target.strip()
        nitpick_ignore.append((dtype, target))


def maybe_skip_member(app, what, name, obj, skip, options):
    """Prevent creating conflicting reference targets for sphinx."""
    deprecated_objs = [
        "quantify_scheduler.operations.stitched_pulse.StitchedPulse",
        "quantify_scheduler.operations.stitched_pulse.convert_to_numerical_pulse",
        "quantify_scheduler.operations.stitched_pulse.StitchedPulseBuilder",
        "quantify_scheduler.operations.pulse_factories.long_ramp_pulse",
        "quantify_scheduler.operations.pulse_factories.long_square_pulse",
        "quantify_scheduler.operations.pulse_factories.staircase_pulse",
    ]
    if str(name) in deprecated_objs:
        return True
    return skip


def setup(app):
    app.connect("autoapi-skip-member", maybe_skip_member)
