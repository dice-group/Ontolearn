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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Ontolearn'
copyright = '2021, The Ontolearn team'
author = 'Ontolearn team'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinxext_autox',
    'sphinx.ext.githubpages',
    # 'sphinx.ext.intersphinx',
    # 'sphinx_automodapi.smart_resolver',
    # 'sphinx.ext.inheritance_diagram',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.plantuml',
    'myst_parser',
    'sphinx_rtd_theme',
]

inheritance_graph_attrs = dict(rankdir="TB")

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

myst_heading_anchors = 3

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
# autosummary_imported_members = True

numpydoc_show_class_members = False
autoclass_content = 'both'

python_use_unqualified_type_names = True
# add_module_names = False

pygments_style = 'rainbow_dash'

plantuml_output_format = 'svg_img'
plantuml_latex_output_format = 'pdf'
# plantuml_batch_size = 100

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

stanford_theme_mod = True
html_theme_options = {
    'navigation_depth': 6,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [
    '_static'
]

if stanford_theme_mod:
    html_theme = 'sphinx_rtd_theme'

    def _import_theme():
        import os
        import shutil
        import sphinx_theme
        html_theme = 'stanford_theme'
        for _type in ['fonts']:
            shutil.copytree(
                os.path.join(sphinx_theme.get_html_theme_path(html_theme),
                             html_theme, 'static', _type),
                os.path.join('_static_gen', _type),
                dirs_exist_ok=True)
        shutil.copy2(
            os.path.join(sphinx_theme.get_html_theme_path(html_theme),
                         html_theme, 'static', 'css', 'theme.css'),
            os.path.join('_static_gen', 'theme.css'),
        )

    _import_theme()
    html_static_path = ['_static_gen'] + html_static_path

# -- Options for LaTeX output ------------------------------------------------

latex_engine = 'xelatex'
latex_show_urls = 'footnote'
# latex_show_pagerefs = True
latex_theme = 'howto'

latex_elements = {
    'preamble': r'''
\renewcommand{\pysiglinewithargsret}[3]{%
  \item[{%
      \parbox[t]{\linewidth}{\setlength{\hangindent}{12ex}%
        \raggedright#1\sphinxcode{(}\linebreak[0]{\renewcommand{\emph}[1]{\mbox{\textit{##1}}}#2}\sphinxcode{)}\linebreak[0]\mbox{#3}}}]}
''',
    'printindex': '\\def\\twocolumn[#1]{#1}\\footnotesize\\raggedright\\printindex',
}


def setup(app):
    # -- Options for HTML output ---------------------------------------------
    if stanford_theme_mod:
        app.add_css_file('theme.css')
    app.add_css_file('theme_tweak.css')
    app.add_css_file('pygments.css')
