# <img src="https://raw.githubusercontent.com/hatchet/hatchet/develop/logo-hex.png" width="64" valign="middle" alt="hatchet"/> Hatchet

[![Build Status](https://github.com/hatchet/hatchet/actions/workflows/unit-tests.yaml/badge.svg)](https://github.com/hatchet/hatchet/actions)
[![Read the Docs](http://readthedocs.org/projects/hatchet/badge/?version=latest)](http://hatchet.readthedocs.io)
[![codecov](https://codecov.io/gh/hatchet/hatchet/branch/develop/graph/badge.svg)](https://codecov.io/gh/hatchet/hatchet)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Join slack](https://img.shields.io/badge/slack-hatchet--users-blue)](https://join.slack.com/t/hatchet-users/shared_invite/zt-twjzzdav-p1s7NUEJzBoejYdOAgeddg)

Hatchet is a Python-based library that allows [Pandas](https://pandas.pydata.org) dataframes to be indexed by structured tree and graph data. It is intended for analyzing performance data that has a hierarchy (for example, serial or parallel profiles that represent calling context trees, call graphs, nested regions’ timers, etc.). Hatchet implements various operations to analyze a single hierarchical data set or compare multiple data sets, and its API facilitates analyzing such data programmatically.

To use hatchet, install it with pip:

```
$ pip install hatchet
```

Or, if you want to develop with this repo directly, run the install script from
the root directory, which will build the cython modules and add the cloned
directory to your `PYTHONPATH`:

```
$ source install.sh
```

<p align="center">
  <img src="https://raw.githubusercontent.com/hatchet/hatchet/develop/screenshot.png" width=800>
</p>


### Documentation

See the [Getting Started](https://hatchet.readthedocs.io/en/latest/getting_started.html) page for basic examples and usage. Full documentation is available in the [User Guide](https://hatchet.readthedocs.io/en/latest/user_guide.html).

Examples of performance analysis using hatchet are available [here](https://hatchet.readthedocs.io/en/latest/analysis_examples.html).

### Contributing

Hatchet is an open source project. We welcome contributions via pull requests,
and questions, feature requests, or bug reports via issues.

You can connect with the hatchet community on
[slack](https://join.slack.com/t/hatchet-users/shared_invite/zt-twjzzdav-p1s7NUEJzBoejYdOAgeddg).
You can also reach the hatchet developers by email at:
[hatchet-help@listserv.umd.edu](mailto:hatchet-help@listserv.umd.edu).

### Authors

Many thanks go to Hatchet's
[contributors](https://github.com/hatchet/hatchet/graphs/contributors).

Hatchet was created by Abhinav Bhatele, bhatele@cs.umd.edu.


### Citing Hatchet

If you are referencing Hatchet in a publication, please cite the
following [paper](http://www.cs.umd.edu/~bhatele/pubs/pdf/2019/sc2019.pdf):

 * Abhinav Bhatele, Stephanie Brink, and Todd Gamblin. Hatchet: Pruning
   the Overgrowth in Parallel Profiles. In Proceedings of the International
   Conference for High Performance Computing, Networking, Storage and Analysis
   (SC '19). ACM, New York, NY, USA. [DOI](
   http://doi.acm.org/10.1145/3295500.3356219)

### License


Hatchet is distributed under the terms of the MIT license.

All contributions must be made under the MIT license.  Copyrights in the
Hatchet project are retained by contributors.  No copyright assignment is
required to contribute to Hatchet.

See [LICENSE](https://github.com/hatchet/hatchet/blob/develop/LICENSE) and
[NOTICE](https://github.com/hatchet/hatchet/blob/develop/NOTICE) for details.

SPDX-License-Identifier: MIT

LLNL-CODE-741008
