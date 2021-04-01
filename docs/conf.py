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
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.intersphinx',
    # 'sphinx_automodapi.smart_resolver',
    'sphinx.ext.autodoc',
    # 'sphinx.ext.inheritance_diagram',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.plantuml',
    'myst_parser',
    'sphinx_rtd_theme',
]

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

def _autosummary_shorten_toc():
    import sphinx.ext.autosummary
    _init = sphinx.ext.autosummary.autosummary_toc.__init__
    def my_init(self, rawsource='', text='', *children, **attributes):
        for tocnode in children:
            tocnode['entries'] = [(_[1].split('.')[-1] if '.' in _[1] else _[0], _[1]) for _ in tocnode['entries']]
        _init(self, rawsource, text, *children, **attributes)

    sphinx.ext.autosummary.autosummary_toc.__init__ = my_init

_autosummary_shorten_toc()

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

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 6,
}

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

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [
    '_static_gen',
    '_static'
]

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

# def maybe_skip_member(app, what, name, obj, skip, options):
#     import inspect
#     from sphinx.util.inspect import safe_getattr
#     mod = safe_getattr(obj, '__module__', None)
#     dp = options.get('documenter_parent', None)
#     dp_all = list(safe_getattr(dp, '__all__', [])) if dp is not None else []
#     if skip:
#         return True
#     if dp is not None and inspect.ismodule(dp) and mod is not None and dp.__name__ != mod and name not in dp_all:
#         # print("SKIPPING[imported] ", [app, what, name, obj, skip, options, dp, mod])
#         return True
#     return None  # ask the next handler

def setup(app):
    # app.connect('autodoc-skip-member', maybe_skip_member)
    # -- Options for HTML output ---------------------------------------------
    app.add_css_file('theme.css')
    app.add_css_file('theme_tweak.css')
    app.add_css_file('pygments.css')
