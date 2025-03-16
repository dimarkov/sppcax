.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/sppcax.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/sppcax
    .. image:: https://readthedocs.org/projects/sppcax/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://sppcax.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/sppcax/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/sppcax
    .. image:: https://img.shields.io/pypi/v/sppcax.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/sppcax/
    .. image:: https://img.shields.io/conda/vn/conda-forge/sppcax.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/sppcax
    .. image:: https://pepy.tech/badge/sppcax/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/sppcax
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/sppcax

.. image:: https://readthedocs.org/projects/sppcax/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://sppcax.readthedocs.io/en/latest/

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

======
sppcax
======


    Sparse Probabilistic Principal Component Analysis with Bayesian Model Reduction written in Jax


sppcax is a Python implementation of sparse probabilistic principal component  analysis (SPPCA) using Bayesian model reduction and coordinate ascent variational inference. This method provides an efficient way to perform dimensionality reduction while automatically determining the optimal number of components and encouraging sparsity in the loading matrix.

Installation
============

You can install sppcax for development with:

.. code-block:: bash

   git clone https://github.com/dimarkov/sppcax.git
   cd sppcax
   pip install -e .

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
