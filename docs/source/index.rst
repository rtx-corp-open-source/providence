Providence |version|
====================


.. include:: ../../README.md
   :parser: myst_parser.sphinx_


Functional Overview
-------------------

See the :doc:`User Guide <userguide>` and `Examples Notebooks <https://github.com/rtx-corp-open-source/providence/tree/main/examples>`_.

Documentation
-------------

You're looking at the documentation.  The docs from the most recent release are hosted on
`github pages <libaxiom.rtxdatascience.com>`_.
When you add new transformers, utilities, or new features or additions, please update and write docstrings/doctests and add them to the docs
directory. To rebuild the docs, you will need to have the python packages sphinx, nbsphinx, myst-parser, ipykernel, and jupyter installed to your
development environment. You will also need to install pandoc, instructions found `here <https://pandoc.org/installing.html>`_.
Once your environment is configured, navigate to the `docs/` directory and run `make clean html`.

Issues
------

If there are any issues, bugs or feature requests, please create an issue on GitHub Issues


Contents:
---------

.. toctree::
   :maxdepth: 3
   :caption: Product Overview

   Providence Overview <self>

.. toctree::
   :maxdepth: 5
   :caption: Data

   research_datasets/index
   data_preparation/index

.. toctree::
   :maxdepth: 5
   :caption: Models

   rnn/index
   transformers/index

.. toctree::
   :maxdepth: 3
   :caption: Activations

   activations/weibull

.. toctree::
   :maxdepth: 3
   :caption: Distributions

   distributions/distributions

.. toctree::
   :maxdepth: 3
   :caption: Extra

   extra/providence_typing
   extra/metrics
   extra/providence_utils
   extra/loss_utils
   extra/transformer_utils
   extra/training_utils
   extra/visualization

.. toctree::
   :maxdepth: 1
   :caption: Examples
   
   examples/examples
   examples/paper_reproduction
   experiments/index

.. toctree::
   :maxdepth: 3
   :caption: Design Philosophy

   design/design_philosophy

.. toctree::
   :maxdepth: 3
   :caption: Legal & Compliance 

   legal/legal


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
