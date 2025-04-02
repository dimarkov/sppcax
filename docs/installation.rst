============
Installation
============

This section provides instructions for installing the ``sppcax`` package.

Requirements
============

``sppcax`` requires the following dependencies:

- Python (>=3.12)
- JAX
- Equinox
- multipledispatch

Stable Release
==============

To install the stable release of ``sppcax`` from PyPI:

.. code-block:: console

    $ pip install sppcax

This is the preferred method to install ``sppcax`` as it will always install the most recent stable release.

From Source
===========

If you want to install the latest development version from the repository:

.. code-block:: console

    $ pip install git+https://github.com/username/sppcax.git

Development Installation
========================

For development purposes, you can clone the repository and install it in development mode:

.. code-block:: console

    $ git clone https://github.com/username/sppcax.git
    $ cd sppcax
    $ pip install -e .

This mode allows you to modify the code and have the changes take effect immediately without reinstalling.

Install Test Dependencies
=========================

To run the tests, additional dependencies are required:

.. code-block:: console

    $ pip install -e ".[testing]"

That will install all the extra dependencies needed for testing.

JAX Installation Notes
======================

JAX installation may require special consideration depending on your hardware:

- For CPU-only installation:

  .. code-block:: console

      $ pip install --upgrade "jax[cpu]"

- For CUDA (NVIDIA GPU) support:

  .. code-block:: console

      $ pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

Please refer to the `JAX installation guide <https://github.com/google/jax#installation>`_ for more details.
