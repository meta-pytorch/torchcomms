%YAML 1.2
---
# Specification and example document for metric taxonomies.

# Each HPCToolkit database provides post-processed performance data for every
# calling context, application thread and performance metric. Performance
# metrics are generally very specific and the impact on the application
# performance is not always clear (eg. is 98% of the GPU L2 misses on a single
# line a problem?).

# Files of this format provide a full "taxonomy" of metrics, structured to aid
# manual performance analysis. Very general metrics (eg. time) are presented
# first to give a sense for *where* significant performance issues are, which
# can be expanded to present increasingly specific metrics to determine the
# *why* and *how*. In other words, the majority of an HPCToolkit database
# (see FORMATS.md) provides raw performance metrics, while METRICS.yaml files
# provide the interpretation.

# This format is primarily intended to be read by the GUI application of
# HPCToolkit, HPCViewer. A number of keys in this file only make sense in this
# context, for instance options on how to present the final metric values.

# NOTE: !!-type specifiers when used below indicate the type(s) allowed for
# the various keys. They are not required and match up with the default type
# as interpreted by most general YAML parsers.

# Version of the METRICS.yaml format required by this file. Can be used by
# readers to error gracefully without reading the entire file. If omitted
# version checks are disabled.
version: !!int 0

# Set of all performance metrics used by this taxonomy. These correspond to
# performance metrics listed in the meta.db file.
# Anchors are used to refer to these metrics later in the file.
inputs: !!seq
  - &in-cycles-E
    # Canonical name for the performance metric.
    # See Performance Metric Specification in FORMATS.md for details.
    metric: !!str perf::cycles
    # Name of the propagation scope for the value referenced.
    # See Performance Metric Specification in FORMATS.md for details.
    scope: !!str function
    # Unary function used to generate summary statistic values, see Performance
    # Metric Specification in FORMATS.md for details.
    # This is a formula in the same format as the variants:formula:* keys in
    # in the metric description below, with the following differences:
    #  - The formula must consist of a single !!str, not a !!seq or other
    #    formula structure ("$$" is used as the variable), and
    #  - The formula is canonicalized: whitespace and extraneous paratheticals
    #    should be removed to achieve a match.
    # Defaults to '$$'.
    formula: !!str $$
    # Combination function use to generate summary statistic values, see
    # Performance Metric Specification in FORMATS.md for details.
    # One of 'sum', 'min' or 'max'. Defaults to 'sum'.
    combine: !!str sum
  # Merge keys can be used to lower the repetition of common fields:
  - &in-cycles-I
    <<: *in-cycles-E
    scope: execution
  - &in-cycles-E-cnt
    <<: *in-cycles-E
    formula: 1
  - &in-cycles-I-cnt
    <<: *in-cycles-I
    formula: 1

  - &in-l1-miss-E
    metric: perf::l1-cache-miss
    scope: function
  - &in-l1-miss-I
    <<: *in-l1-miss-E
    scope: execution
  - &in-l1-miss-E-cnt
    <<: *in-l1-miss-E
    formula: 1
  - &in-l1-miss-I-cnt
    <<: *in-l1-miss-I
    formula: 1

  - &in-l2-miss-E
    metric: perf::l2-cache-miss
    scope: function
  - &in-l2-miss-I
    <<: *in-l2-miss-E
    scope: execution
  - &in-l2-miss-E-cnt
    <<: *in-l2-miss-E
    formula: 1
  - &in-l2-miss-I-cnt
    <<: *in-l2-miss-I
    formula: 1

  - &in-l3-miss-E
    metric: perf::l3-cache-miss
    scope: function
  - &in-l3-miss-I
    <<: *in-l3-miss-E
    scope: execution
  - &in-l3-miss-E-cnt
    <<: *in-l3-miss-E
    formula: 1
  - &in-l3-miss-I-cnt
    <<: *in-l3-miss-I
    formula: 1

# Sequence of root metrics provided in this taxonomy. Every metric listed in the
# taxonomy is a descendant of one of these.
roots:
  - # Name for the metric.
    name: !!str CPU Cycles
    # Longer description of the metric, written in Markdown.
    # Defaults to the `short description:` if given.
    description: >
      Cycles spent:
       - In the CPU doing actual work (FLOPs), or
       - Waiting for outside operations to complete (memory stalls).
    # Short description of the metric, used for cases where a long description
    # would not be suitable.
    # Defaults to `description:` up to the first period or newline.
    short description: !!str Cycles spent in the CPU.

    # Whether this metric should be visible in the Viewer by default, default
    # true. If false, the Viewer may require that the metric be enabled in the
    # metric list before it will be presented.
    visible by default: true

    # How the values in the metrics rooted here will be presented in the Viewer
    # by default. One of:
    #  - 'column': Columns of data that can be expanded to show inner metrics.
    # Defaults to 'column'. Only allowed on root metrics.
    presentation: !!str column

    # Sequence of child metrics, format is the same as a root metric.
    # If omitted there are no child metrics.
    children: !!seq
    - name: L2 Bound
      description: Rough cycles spent accessing the L2 cache

      # List of formula variations for this taxonomic metric. Metric values are
      # always attributed to an application thread, however for large executions
      # this gives too much data to present clearly. Instead, the Viewer
      # presents on "summary" values by applying statistics across threads.
      # The `inputs:` key above lists the "partial" results required for
      # calculating statistics, this key lists the final formulas to generate
      # presentable values.
      #
      # Keys in this map are the human-readable names of the variants.
      variants: !!map
        !!str Sum:
          # How the final value(s) for this metric variant should be rendered.
          # Orderless set of elements to be rendered in the metric cell, the
          # following options are available:
          #  - 'number': Numerical rendering (see `format:`).
          #  - 'percentage': Percentage of the global inclusive value. Only
          #    allowed if `formula:inclusive:` is given.
          #  - 'hidden': Mark as hiding (some) inner values (`*`).
          #  - 'colorbar': Color bar visually indicating the relative sizes of
          #    values in child metrics. An additional "grey" color is added to
          #    the bar to indicate the difference between sum-of-children and
          #    this metric variant's value. (Note that this difference will be
          #    exactly 0 if `formula:` is 'sum'.)
          # The Viewer will order the elements reasonably, and may elide
          # elements if screen real estate is tight.
          render: !!seq [number, percent]  # eg: 1.23e+04 56.7%
          # Can also be given as a !!str for a single element:
          render: !!str 'number'  # eg: 1.23e+04

          # Printf-like format to use when rendering the metric value(s) as a
          # number (`render: number`). The input to "printf" is a single double
          # value. Defaults to '%.2e'.
          #
          # In more detail, this string must be of the form:
          #     [prefix]%(#0- +')*[field width][.precision](eEfFgGaA)[suffix]
          # Where "prefix" and "suffix" use %% to generate a literal %.
          format: !!str '%.2e'

          # Which variant child metric values are gotten from. Also used as the
          # default variant when first expanding this metric variant. Explicitly
          # lists the variant to use for each child metric in order.
          child variant: !!seq
            - Sum   # Use Sum value(s) from first child
            - Mean  # Use Mean value(s) from second child
          # Or can also be given as a !!str if the variant is the same.
          child variant: !!str Sum  # Use Sum value(s) from all children
          # Defaults to the name of this variant.

          # Formula(s) for calculating the final value(s) for this metric
          # variant. Ignored unless `render:` contains a numerical element
          # (ie. everything except 'hidden'). Can be one of:
          #  - 'first': Value(s) for this variant are copied from the value(s)
          #    of the first child. Invalid if `render:` contains 'colorbar'.
          #  - 'sum': Value(s) are generated by summing child value(s).
          # In all cases value(s) are generated vector-wise (ie. inclusive
          # values come from inclusive child values, exclusive from exclusive,
          # etc.), and null child values generate null values in the parent
          # (ie. they aren't replaced with 0).
          formula: !!str first
          # Can also be written as a !!map listing the vector of formulas.
          formula: !!map
            # The following keys define the formulas used to generate metrics.
            # Formulas are roughly written as a C-like math expression, except:
            # - "Variables" are references to other nodes, which can be other
            #   formulas (sub-expressions) or an entry in the global `inputs:`.
            #   Eg: `*in-cycles-E` is an input metric value.
            # - Parentheses are represented with a YAML !!seq ([...]), breaks
            #   between elements (,) are considered whitespace.
            #   Eg: `2 * (3 + 4)` -> `[2 *,[3,+,4]]`
            # - Number constants and infix operators can be represented by
            #   !!int, !!float and !!str YAML elements (as appropriate), and
            #   need not be separated by an element break (whitespace suffices).
            #   Eg: `[2 *,[3,+,4]]` == `[2,*,[3+4]]`
            #   The following operators are available in increasing precedence:
            #       + -    # Addition and subtraction
            #       * /    # Multiplication and (true) division
            #       ^      # Exponentiation
            # - Function calls are represented by a YAML !!map with a single
            #   pair. The key is the function name and the value is a !!seq
            #   listing the arguments.
            #   Eg: `foo(1, 2, 3)` -> `[foo:[1,2,3]]`,
            #       and `foo(1+x)` -> `[foo:[ [1+,*x] ]]`
            #   The following functions are available:
            #       sum:[...]    # Sum of arguments
            #       prod:[...]   # Product of arguments
            #       pow:[a, b]   # a raised to the b
            #       sqrt:[a]     # Square root of a (pow(a, .5))
            #       log:[a, b]   # Logarithm of a base-b
            #       log:[a]      # Natural logarithm of a
            #       min:[...]    # Smallest of arguments
            #       max:[...]    # Largest of arguments
            #       floor:[a]    # Largest integer less than or equal to a
            #       strict floor:[a]    # Largest integer less than a
            #       ceil:[a]     # Smallest integer greater than or equal to a
            #       strict ceil:[a]    # Smallest integer greater than a

            # Formulas to generate "inclusive" cost values. Defaults to null.
            inclusive:
              # Custom formula used when no special properties are required of
              # the formulation. Defaults to the value of `standard:`.
              custom: [4*,[*in-l1-miss-I,-,*in-l2-miss-I]]

              # Version of the formula based completely on well-defined metric
              # inputs, which refer only to non-custom propagation scopes. Used
              # in the bottom-up and flat views, where this property is required
              # for accurate analysis. Defaults to null.
              # See the meta.db Performance Metrics section for details.
              standard: [4*,[*in-l1-miss-I,-,*in-l2-miss-I]]

            # Formulas to generate "exclusive" cost values. Defaults to null.
            exclusive:
              standard: [4*,[*in-l1-miss-E,-,*in-l2-miss-E]]

        # Another example variant for "L2 Bound"
        Mean:
          render: [number, percent]
          formula:
            inclusive: [4*,[*in-l1-miss-I,/,*in-l1-miss-I-cnt, -,*in-l2-miss-I,/,*in-l2-miss-I-cnt]]
            exclusive: [4*,[*in-l1-miss-E,/,*in-l1-miss-E-cnt, -,*in-l2-miss-E,/,*in-l2-miss-E-cnt]]

    # Sibling metric, still under "CPU Cycles"
    - name: L3 Bound
      description: Rough cycles spent accessing L3 cache
      variants:
        Sum:
          render: number
          formula:
            inclusive: [64*,[*in-l2-miss-I, -,*in-l3-miss-I]]
            exclusive: [64*,[*in-l2-miss-E, -,*in-l3-miss-E]]
        Mean:
          render: [number, percent]
          formula:
            inclusive: [64*,[*in-l2-miss-I,/,*in-l2-miss-I-cnt, -,*in-l3-miss-I,/,*in-l3-miss-I-cnt]]
            exclusive: [64*,[*in-l2-miss-E,/,*in-l2-miss-E-cnt, -,*in-l3-miss-E,/,*in-l3-miss-E-cnt]]

    # Parameters for the root "CPU Cycles" metric
    variants:
      Sum:
        render: number
        formula:
          inclusive: *in-cycles-I
          exclusive: *in-cycles-E
      Mean:
        render: [number, colorbar]
        formula:
          inclusive: [*in-cycles-I,/,*in-cycles-I-cnt]
          exclusive: [*in-cycles-E,/,*in-cycles-E-cnt]
