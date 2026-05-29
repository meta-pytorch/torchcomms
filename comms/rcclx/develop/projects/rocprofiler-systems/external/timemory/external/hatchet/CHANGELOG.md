# v1.3.1a0 (2021-04-02)

This release extends the query language to support querying nodes that
fall within a range and includes a critical fix for using the query language on
a dataframe that may contain nan and infs values.

### New features
* Timemory reader
* Query dataframe columns with GraphFrame.show_metric_columns()
* Query nodes within a range using the call path query language
* Extend readers to define their own default metric

### Changes to existing APIs
* Tree visualization displays 2 metrics
* Literal output format: add hatchet node IDs
* Parallel implementationo of filter function
* Caliper reader: support multiple hierarchies in JSON format
* Adds multiprocessing dependency

### Bugfixes
* Improve querying of subtrees in interactive visualization
* Tree visualization: check for both nan and inf
* Query language: match nodes with nan and inf values
* Check for exclusive metrics before updating inclusive columns

# v1.3.0 (2020-11-12)

This release introduces a new tree visualization for Jupyter to interact with
the data, deprecates some of the tree parameters, adds cython as a dependency,
and contains performance improvements to two of Hatchet's central APIs.

### New features

* Interactive tree visualization in Jupyter
* Add mult and division API
* Update hatchet installation steps for cython integration
* Readers: cprofiler, pyinstrument
* Graph output formats: to_literal
* Add profiling APIs to profile Hatchet APIs
* Update basic tutorial for hatchet

### Changes to existing APIs

* Remove threshold=, color=, and unicode= from tree API
* Highlighting name disabled by default in terminal tree output
  is kept in sync with the dataframe
* Internal performance improvements to unify and HPCToolkit reader, enabling
  analysis of large datasets
* For mathematical operations, insert nan values for missing nodes, show values
  as nan and inf as necessary in dataframe
* Extend callpath query language to support non-dataframe metrics (e.g., depth,
  hatchet ID)
* Literal reader: A node can be defined with a "duplicate": True field if it
  should be the same node (though in a different callpath). A node also needs
  "frame" field, which is a dict containing the node "name" and "type" (if
  necessary).

### Bugfixes

* caliper reader: do not assume "path" column is present, create graph after
  reading metadata
* fix updating inclusive columns for multi-indexed dataframes (required as part
  of squash)
* fix unicode error in python2 for terminal-based tree output

# v1.2.0 (2020-07-07)

This release adds a syntax query language as an alternative method for
filtering the graph. It also refreshes the tree printout with an updated
format and legend.

### New features

* Add graph syntax query language to filter the graph
* Update HPCToolkit reader to handle sec or usec time units

### Changes to existing APIs

* Add squash parameter to filter function to perform filter and squash in a
  single call
* Filter function takes an object, which can be a user-supplied function or a
  query object
* Tree printout format updated
* Tree printout API parameter changes:
    - Removed parameters: ``color`` deprecated (color determined based on terminal support),
      ``threshold``, and ``unicode``
    - Changed parameters: ``metric`` changed to ``metric_column``, ``name`` changed to
      ``name_column``, ``invert_colors`` changed to ``invert_colormap``,
      ``expand_names`` changed to ``expand_name``, and ``context`` changed to
      ``context_column``
    - added ``highlight_name`` to highlight user code (from non-user code)

### Bugfixes

* Sort nodes in union and tree printout by their frame
* Fix squash edge case where multiple parents are the same

# v1.1.0 (2020-05-07)

This release adds new analysis operators, as well as some bugfixes and minor
changes.

### New analysis operations

* Add GraphFrame reindex operator
* Query hatchet module version

### Changes to existing APIs

* Add depth parameter to tree printout

### Bugfixes

* Fix pandas SettingwithCopyWarning in unify's _missing_nodes
* Handle MultiIndex for pandas 1.0.0 and newer vs older pandas versions

# v1.0.1 (2020-03-11)

This release adds a new division operator and graph markups, as well as
bugfixes and minor changes.

### New analysis operations

* Add markups to identify nodes that exist in only one of two graphs (from unify)
* Add GraphFrame division operator

### Changes to existing APIs

* Add precision parameter (of metrics) in tree printout
* Tree printout to show nodes with negative values higher than threshold

### Bugfixes

* Fix HPCToolkit reader bug for statement nodes
* Downgrade pandas version for python 3.6 and later (incompatible versions)
* Fix unify by adding missing rows for math operations on GraphFrames
* Fix squash by restoring index in self's dataframe
* Do not sort nodes by frame in Graph union
* Fix phase timer to aggregate times for duplicate phases
* Remove node callpath calculation from HPCToolkit reader
* Remove unnecessary setting of _hatchet_nid in dataframe

# v1.0.0 (2019-11-18)

`v1.0.0` marks the first release of Hatchet!

### Analysis operations

* File formats supported: HPCToolkit, Caliper, DOT, string literal, list
* Graph visualization formats: terminal output, DOT, flamegraph
* Analysis operations: filter, squash, add, subtract, unify, copy

### Testing and Documentation

* Hatchet added to PyPI repository
* Unit tests using `pytest`
* Initial documentation on [hatchet.readthedocs.io](http://hatchet.readthedocs.io)
