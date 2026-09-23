.. Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
   Hatchet Project Developers. See the top-level LICENSE file for details.

   SPDX-License-Identifier: MIT

.. hatchet documentation master file, created by
   sphinx-quickstart on Tue Jun 26 08:43:21 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#######
Hatchet
#######

Hatchet is a Python-based library that allows `Pandas
<https://pandas.pydata.org>`_ dataframes to be indexed by structured tree and
graph data. It is intended for analyzing performance data that has a hierarchy
(for example, serial or parallel profiles that represent calling context trees,
call graphs, nested regions' timers, etc.).  Hatchet implements various
operations to analyze a single hierarchical data set or compare multiple data
sets, and its API facilitates analyzing such data programmatically.

You can get hatchet from its `GitHub repository
<https://github.com/hatchet/hatchet>`_:

.. code-block:: console

  $ git clone https://github.com/hatchet/hatchet.git

or install it using pip:

.. code-block:: console

  $ pip install hatchet

If you are new to hatchet and want to start using it, see :doc:`Getting Started
<getting_started>`, or refer to the full :doc:`User Guide <user_guide>` below.


.. toctree::
   :maxdepth: 2
   :caption: User Docs

   getting_started
   user_guide
   query_lang
   data_generation
   analysis_examples

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   basic_tutorial


If you encounter bugs while using hatchet, you can report them by opening an issue on `GitHub <http://github.com/hatchet/hatchet/issues>`_.

If you are referencing hatchet in a publication, please cite the
following `paper <http://www.cs.umd.edu/~bhatele/pubs/pdf/2019/sc2019.pdf>`_:

* Abhinav Bhatele, Stephanie Brink, and Todd Gamblin. Hatchet: Pruning
  the Overgrowth in Parallel Profiles. In Proceedings of the International
  Conference for High Performance Computing, Networking, Storage and Analysis
  (SC '19). ACM, New York, NY, USA.
  `DOI <https://doi.org/10.1145/3295500.3356219>`_

.. toctree::
   :maxdepth: 2
   :caption: Developer Docs

   developer_guide

.. toctree::
   :maxdepth: 2
   :caption: References

   publications

.. toctree::
   :maxdepth: 2
   :caption: API Docs

   Hatchet API Docs <source/hatchet>


##################
Indices and tables
##################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
