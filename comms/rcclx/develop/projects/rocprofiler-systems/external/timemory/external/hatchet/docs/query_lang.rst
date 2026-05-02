.. Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
   Hatchet Project Developers. See the top-level LICENSE file for details.

   SPDX-License-Identifier: MIT

**************
Query Language
**************

As of version 1.2.0, Hatchet has a filtering query language that allows users to filter GraphFrames based on caller-callee relationships between nodes in the Graph. This query language contains two APIs: a high-level API that is expressed using built-in Python data types (e.g., lists, dictionaries, strings) and a low-level API that is expressed using Python callables. 

Regardless of API, queries in Hatchet represent abstract paths, or path patterns, within the Graph being filtered. When filtering on a query, Hatchet will identify all paths in the Graph that match the query. Then, it will return a new GraphFrame object containing only the nodes contained in the matched paths. A query is represented as a list of *abstract graph nodes*. Each *abstract graph node* is made of two parts:

- A wildcard that specifies the number of real nodes to match to the abstract node
- A filter that is used to determine whether a real node matches the abstract node

The primary differences between the two APIs are the representation of filters, how wildcards and filters are combined into *abstract graph nodes*, and how *abstract graph nodes* are combined into a full query.

The following sections will describe the specifications for queries in both APIs and provide examples of how to use the query language.

High-Level API
==============

The high-level API for Hatchet's query language is designed to allow users to quickly write simple queries. It has a simple syntax based on built-in Python data types (e.g., lists, dictionaries, strings). The following subsections will describe each component of high-level queries. After creating a query, it can be used to filter a GraphFrame by passing it to the :code:`GraphFrame.filter` function as follows:

.. code-block:: python

  query = <QUERY GOES HERE>
  filtered_gf = gf.filter(query)

Wildcards
~~~~~~~~~

Wildcards in the high-level API are specified by one of four possible values:

- The string :code:`"."`, which means "match 1 node"
- The string :code:`"*"`, which means "match 0 or more nodes"
- The string :code:`"+"`, which means "match 1 or more nodes"
- An integer, which means "match exactly that number of nodes" (integer 1 is equivalent to :code:`"."`)

Filters
~~~~~~~

Filters in the high-level API are specified by Python dictionaries. These dictionaries are keyed on the names of *node attributes*. These attributes' names are the same as the column names from the DataFrame associated with the GraphFrame being filtered (which can be obtained with :code:`gf.dataframe`). There are also two special attribute names:

- `depth`, which filters on the depth of the node in the Graph
- `node_id`, which filters on the node's unique identifier within the GraphFrame

The values in a high-level API filter dictionary define the conditions that must be passed to pass the filter. Their data types depend on the data type of the corresponding attribute. The table below describes what value data types are valid for different attribute data types.

+----------------------------+--------------------------+------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| Attribute Data Type        | Example Attributes       | Valid Filter Value Types                                                                       | Description of Condition                                                                                       |
+============================+==========================+================================================================================================+================================================================================================================+
| Real (integer or float)    | `time`                   | Real (integer or float)                                                                        | Attribute value exactly equals filter value                                                                    |
+                            +                          +------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
|                            | `time (inc)`             | String starting with comparison operator                                                       | Attribute value must pass comparison described in filter value                                                 |
+----------------------------+--------------------------+------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
| String                     | `name`                   | Regex String (see `Python re module <https://docs.python.org/3/library/re.html>`_ for details) | Attribute must match filter value (passed to `re.match <https://docs.python.org/3/library/re.html#re.match>`_) |
+----------------------------+--------------------------+------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+

The values in a high-level API filter dictionary can also be iterables (e.g., lists, tuples) of the valid values defined in the table above.

In the high-level API, all conditions (key-value pairs, including conditions contained in a list value) in a filter must pass for the a real node to match the corresponding *abstract graph node*.

Abstract Graph Nodes
~~~~~~~~~~~~~~~~~~~~

In the high-level API, *abstract graph nodes* are represented by Python tuples containing a single wildcard and a single filter. Alternatively, an *abstract graph node* can be represented by only a single . When only providing a wildcard or a filter (and not both), the default is used for the other component. The defaults are as follows:

- Wildcard: :code:`"."` (match 1 node)
- Filter: an "always-true" filter (any node passes this filter)

Full Queries
~~~~~~~~~~~~

In the high-level API, a query is represented as a Python list of *abstract graph nodes*. In general, the following code can be used as a template to build a low-level query.

.. code-block:: python

   query = [
       (wildcard1, query1),
       (wildcard2, query2),
       (wildcard3, query3)
   ]
   filtered_gf = gf.filter(query)

Low-Level API
=============

The low-level API for Hatchet's query language is designed to allow users to perform more complex queries. It's syntax is based on Python callables (e.g., functions, lambdas). The following subsections will describe each component of low-level queries. Like high-level queries, low-level queries can be used to filter a GraphFrame by passing it to the :code:`GraphFrame.filter` function as follows:

.. code-block:: python

  query = <QUERY GOES HERE>
  filtered_gf = gf.filter(query)

Wildcards
~~~~~~~~~

Wildcards in the low-level API are the exact same as wildcards in the high-level API. The following values are currently allowed for wildcards:

- The string :code:`"."`, which means "match 1 node"
- The string :code:`"*"`, which means "match 0 or more nodes"
- The string :code:`"+"`, which means "match 1 or more nodes"
- An integer, which means "match exactly that number of nodes" (integer 1 is equivalent to :code:`"."`)

Filters
~~~~~~~

The biggest difference between the high-level and low-level APIs are how filters are represented. In the low-level API, filters are represented by Python callables. These callables should take one argument representing a node in the graph and should return a boolean stating whether or not the node satisfies the filter. The type of the argument to the callable depends on whether the :code:`GraphFrame.drop_index_levels` function was previously called. If this function was called, the type of the argument will be a :code:`pandas.Series`. This :code:`Series` will be the row representing a node in the internal :code:`pandas.DataFrame`. If the :code:`GraphFrame.drop_index_levels` function was not called, the type of the argument will be a :code:`pandas.DataFrame`. This :code:`DataFrame` will contain the rows of the internal :code:`pandas.DataFrame` representing a node. Multiple rows are returned in this case because the internal :code:`DataFrame` will contain one row for every thread and function call.

For example, if you want to match nodes with an exclusive time (represented by "time" column) greater than 2 and an inclusive time (represented by "time (inc)" column) greater than 5, you could use the following filter. This filter assumes you have already called the :code:`GraphFrame.drop_index_levels` function.

.. code-block:: python

   filter = lambda row: row["time"] > 2 and row["time (inc)"] > 5

Abstract Graph Nodes
~~~~~~~~~~~~~~~~~~~~

To build *abstract graph nodes* in the low-level API, you will first need to import Hatchet's :code:`QueryMatcher` class. This can be done with the following import.

.. code-block:: python

   from hatchet import QueryMatcher

The :code:`QueryMatcher` class has two functions that can be used to build *abstract graph nodes*. The first function is :code:`QueryMatcher.match`, which resets the query and constructs a new *abstract graph node* as the root of the query. The second function is :code:`QueryMatcher.rel`, which constructs a new *abstract graph node* and appends it to the query. Both of these functions take two arguments: a wildcard and a low-level filter. If either the filter or wildcard are not provided, the default will be used. The defaults are as follows:

- Wildcard: :code:`"."` (match 1 node)
- Filter: an "always-true" filter (any node passes this filter)

Both of these functions also return a reference to the :code:`self` parameter of the :code:`QueryMatcher` object. This allows :code:`QueryMatcher.match` and :code:`QueryMatcher.rel` to be chained together.

Full Queries
~~~~~~~~~~~~

Full queries in the low-level API are built by making sucessive calls to the :code:`QueryMatcher.match` and :code:`QueryMatcher.rel` functions. In general, the following code can be used as a template to build a low-level query.

.. code-block:: python

   from hatchet import QueryMatcher

   query = QueryMatcher().match(wildcard1, filter1)
       .rel(wildcard2, filter2)
       .rel(wildcard3, filter3)
   filtered_gf = gf.filter(query)

Compound Queries
================

*Compound queries is currently a development feature.*

Compound queries allow users to apply some operation on the results of one or more queries. Currently, the following compound queries are available directly from :code:`hatchet.query`:

- :code:`AndQuery` and :code:`IntersectionQuery`
- :code:`OrQuery` and :code:`UnionQuery`
- :code:`XorQuery` and :code:`SymDifferenceQuery`

Additionally, the compound query feature provides the following abstract base classes that can be used by users to implement their own compound queries:

- :code:`AbstractQuery`
- :code:`NaryQuery`

The following subsections will describe each of these compound query classes.

AbstractQuery
~~~~~~~~~~~~~

:code:`AbstractQuery` is an interface (i.e., abstract base class with no implementation) that defines the basic requirements for a query in the Hatchet query language. All query types, including user-created compound queries, must inherit from this class.

NaryQuery
~~~~~~~~~

:code:`NaryQuery` is an abstract base class that inherits from :code:`AbstractQuery`. It defines the basic functionality and requirements for compound queries that perform one or more subqueries, collect the results of the subqueries, and performs some subclass defined operation to merge the results into a single result. Queries that inherit from :code:`NaryQuery` must implment the :code:`_perform_nary_op` function, which takes a list of results and should perform some operation on it.

AndQuery
~~~~~~~~

The :code:`AndQuery` class can be used to perform two or more subqueries and compute the intersection of all the returned lists of matched nodes. To create an :code:`AndQuery`, simply create your subqueries (which can be high-level, low-level, or compound), and pass them to the :code:`AndQuery` constructor. The following code can be used as a template for creating an :code:`AndQuery`.

.. code-block:: python

   from hatchet.query import AndQuery

   query1 = <QUERY GOES HERE>
   query2 = <QUERY GOES HERE>
   query3 = <QUERY GOES HERE>
   and_query = AndQuery(query1, query2, query3)
   filtered_gf = gf.filter(and_query)

:code:`IntersectionQuery` is also provided as an alias (i.e., renaming) of :code:`AndQuery`. The two can be used interchangably.

OrQuery
~~~~~~~~

The :code:`OrQuery` class can be used to perform two or more subqueries and compute the union of all the returned lists of matched nodes. To create an :code:`OrQuery`, simply create your subqueries (which can be high-level, low-level, or compound), and pass them to the :code:`OrQuery` constructor. The following code can be used as a template for creating an :code:`OrQuery`.

.. code-block:: python

   from hatchet.query import OrQuery

   query1 = <QUERY GOES HERE>
   query2 = <QUERY GOES HERE>
   query3 = <QUERY GOES HERE>
   or_query = OrQuery(query1, query2, query3)
   filtered_gf = gf.filter(or_query)

:code:`UnionQuery` is also provided as an alias (i.e., renaming) of :code:`OrQuery`. The two can be used interchangably.

XorQuery
~~~~~~~~

The :code:`XorQuery` class can be used to perform two or more subqueries and compute the symmetric difference (set theory equivalent to XOR) of all the returned lists of matched nodes. To create an :code:`XorQuery`, simply create your subqueries (which can be high-level, low-level, or compound), and pass them to the :code:`XorQuery` constructor. The following code can be used as a template for creating an :code:`XorQuery`.

.. code-block:: python

   from hatchet.query import XorQuery

   query1 = <QUERY GOES HERE>
   query2 = <QUERY GOES HERE>
   query3 = <QUERY GOES HERE>
   xor_query = XorQuery(query1, query2, query3)
   filtered_gf = gf.filter(xor_query)

:code:`SymDifferenceQuery` is also provided as an alias (i.e., renaming) of :code:`XorQuery`. The two can be used interchangably.
