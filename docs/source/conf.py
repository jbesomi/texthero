# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import matplotlib

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "texthero"
copyright = ""  # will not be used.
author = ""  # will not be used.

# The full version, including alpha/beta/rc tags
release = ""  # will not be used.


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",  # automatically construct the documentation.
    "sphinx.ext.autosummary",
    # prefer numpydoc at sphinx.ext.napoleon as it looks nicer.
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*.md"]

add_module_names = False

autosummary_generate = True

autodoc_typehints = "none"


intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"  # "alabaster", "pydata_sphinx_theme"

# html_theme_options = {"nosidebar": "true"}

# html_use_index = False  # Create an extra page containing the index.

# html_show_sourcelink = False

# html_file_suffix = ".md" later

# html_show_copyright = False

# html_show_sphinx = False

# html_domain_indices = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


html_css_files = [
    "css/pigments.css",
    "css/custom.css",
]

autodoc_typehints = "none"

source_suffix = [".rst"]
