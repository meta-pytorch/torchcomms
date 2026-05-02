A full HPCToolkit database consists of the following files and directories:

    database/
    |-- FORMATS.md     This file
    |-- metrics/       Taxonomic metric descriptions for analysis presentations
    |   |-- METRICS.yaml.ex  Documentation for the metric taxonomy YAML format
    |   `-- default.yaml     Default metric taxonomy, suitable for most cases
    |-- meta.db        Properties of the measured application execution
    |-- profile.db     Performance measurements arranged by application thread
    |-- cct.db         Performance measurements arranged by calling context
    |-- trace.db       Time-centric execution traces
    `-- src/           Relevant application source files

This file describes the format of the `*.db` files in an HPCToolkit database.
See `metrics/METRICS.yaml.ex` for a description of the format for defining
performance metric taxonomies.

Table of contents:
  - [Common properties for all formats (READ FIRST)](#common-properties-for-all-formats-read-first)
  - [`meta.db` v4.0](#metadb-version-40)
  - [`profile.db` v4.0](#profiledb-version-40)
  - [`cct.db` v4.0](#cctdb-version-40)
  - [`trace.db` v4.0](#tracedb-version-40)

* * *

Common properties for all formats (READ FIRST)
==============================================

### Formats legend ###
[Formats legend]: #formats-legend

All `*.db` formats are custom binary formats comprised of structures at various
positions within the file. These structures are described by tables of the
following form, where each row (except the first and last) describe a field in
the structure in a notation similar to C's `struct`:

 Hex | Type | Name | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---- | ---- | ----------------------------------------------------
`A 4`|| **ALIGNMENT**    || See [Alignment properties]
`00:`|Ty|`field1`     |4.0| Description of the value in `field1`
|    |
`13:`|Ty2|`field2`    |4.1| Description of the value in `field2`
`15:`|| **END**          || Extendable, see [Reader compatibility]

 - The initial **ALIGNMENT** row indicates the minimum alignment of the absolute
   file offset of the beginning of the structure. For instance, `A 4` in the
   above table indicates that `field1` is placed on a 4-byte boundary within the
   file. See [Alignment properties] for more details.

 - **Hex** lists the constant byte-offset from the beginning of the structure to
   the beginning of the field, in hexadecimal. Fields are normally packed with
   no padding in-between, if there may be a gap between fields an empty row is
   inserted into the table for readability. If the offset is not constant for a
   field no value is listed.

 - **Type** lists the interpretation of the field's bytes, along with
   (implicitly) its size. The standard types are as follows:
   + u`N` is a unsigned integer of `N` bits.
     Multi-byte integers are laid out in little-endian order.
   + f64 is an IEEE 754 double-precision floating-point number, laid out in
     little-endian order (ie. sign byte comes last).
   + `Ty`[`N`] is an array of `N` elements of type `Ty`. There is no padding
     between elements (ie. the stride is equal to the total size of `Ty`).
     `N` may refer to a sibling field, or may be `...` if the size is defined
     in the **Description**.
   + `Ty`* is a u64, but additionally the value is the absolute byte-offset of a
     structure of type `Ty` (ie. a pointer to `Ty`). The value is aligned to the
     minimum alignment of `Ty`, as defined in [Alignment properties].
   + char* is a u64, but additionally the value is the absolute byte-offset of
     the start byte of a null-terminated UTF-8 string. The value is generally
     unaligned.

 - **Name** gives a short name to refer to the field in further descriptions.

 - **Ver.** lists the first version the field first appeared. Note that if the
   offset of the field changed over a major version, this number is will not be
   updated to match the new major version.

 - **Description** describes the value of the field. Longer and additional
   descriptions are listed after the table in separate paragraphs or lists.

 - The final **END** row lists the total size of the structure. If this
   structure is used in an array, this is the offset of the first field in the
   following array element. The **Description** indicates whether the structure
   may be modified in later minor versions (expandable) or not (fixed), see
   [Reader compatibility] for more details.


### Common file structure ###
[Common file structure]: #common-file-structure

All `*.db` files are structured as a file header containing format identifiers
and version information, and references to "sections" in the rest of the file.
The file header always has the following structure:

 Hex | Type | Name     | Description (see the [Formats legend])
 ---:| ---- | -------- | -------------------------------------------------------
`00:`|u8[10]|`magic`   | Common format identifier, reads `HPCTOOLKIT` in ASCII
`0a:`|u8[4]|`format`   | Specific format identifier
`0e:`|u8|`majorVersion`| Common major version, currently 4
`0f:`|u8|`minorVersion`| Specific minor version
`10:`|u64|`szSection1` | Total size of section 1
`18:`|u8*|`pSection1`  | Pointer to the beginning of section 1
`20:`|u64|`szSection2` | Total size of section 2
`28:`|u8*|`pSection2`  | Pointer to the beginning of section 2
`30:`|   | ...etc...   |

`majorVersion` and `minorVersion` indicate the version of the writer of the
file, see [Reader compatibility] for implications.

`format` identifies the specific format for the file, and always reads as a
4-character ASCII string (no terminator). Specifically:
  - `meta` for [`meta.db` v4.0](#metadb-version-40)
  - `prof` for [`profile.db` v4.0](#profiledb-version-40)
  - `ctxt` for [`cct.db` v4.0](#cctdb-version-40)
  - `trce` for [`trace.db` v4.0](#tracedb-version-40)

Additional notes:
 - The structure of file headers, including the value for `magic`, does not
   change across major versions.
 - The values for `format` follow the same rules as enumerations as defined in
   [Reader compatibility].
 - `majorVersion` is consistent across all `*.db` files in one database.
   `minorVersion` is not in general.

The remainder of the file header is made up of pointers of and sizes of
contiguous regions of the file, hereafter termed sections. Many but not all
structures reside in a section. For simplicity, the top-level file header for
each format is written with a shorthand form of the structure table:

 Hex | Name        | Ver. | Section (see the [Common file structure])
 ---:| ----------- | ---- | ----------------------------------------------------
`00:`|                   || See [Common file structure]
`10:`|`{sz,p}Section1`|4.0| Short description of section 1
`20:`|`{sz,p}Section2`|4.2| Short description of section 2
`30:`| ...etc...

The names `Section1` and `Section2` are replaced with more descriptive
identifiers in practice.

Each section generally starts with a section header structure, this indicates
what is in the section and where it is located. Sections are generally grouped
based on related information, and generally contain little padding to facilitate
reading a large blob of related information.

Additional notes:
 - `p*` fields are aligned based on the alignment of the section header
   structure, unless otherwise noted.
 - The order of sections in the file and the order they are listed in the header
   may differ, do not rely on any section ordering properties without checking
   first.


### Alignment properties ###
[Alignment properties]: #alignment-properties

The `*.db` formats are designed for efficient access through memory-mapped
segments, to support this fields are aligned to improve access for
performance-critical readers. All types have a minimum alignment that is
respected (unless otherwise noted), defined as follows:

 - Integers (u`N`) have a minimum alignment equal to their width (`N/8`).
   For example, u32 is 4-byte aligned.
 - Floating point numbers (f64) are 8-byte aligned.
 - Arrays (`Ty`[`N`]) have the same alignment as their elements (`Ty`). In this
   case the total size of `Ty` is a multiple of the alignment of `Ty`, so there
   is no implicit padding between elements.
 - Pointers (`Ty`* and char*) are 8-byte aligned (same as u64).
 - Structures listed with structure tables have the alignment listed in their
   initial **ALIGNMENT** row. In general this is at least the alignment of all
   contained fields.

Note that 8-byte alignment is the maximum possible alignment.

The following fields are not always aligned, see the notes in their defining
sections for recommendations on how to achieve performance in these cases:
 - Performance data arrays in [Profile-Major][PSVB] and
   [Context-Major Sparse Value Block][CSVB], and
 - The array in a [trace line][THsec].

[PSVB]: #profile-major-sparse-value-block
[CSVB]: #context-major-sparse-value-block
[THsec]: #tracedb-trace-headers-section


### Reader compatibility ###
[Reader compatibility]: #reader-compatibility

The `*.db` formats are also designed for high compatibility between readers and
writers as both continue to update. Readers are able to determine the level of
compatibility needed or available by inspecting the major and minor version
numbers in the [Common file structure]. Specifically, we define two kinds of
compatibility, taken from the reader's perspective:
 - *Backward compatibility*, when the reader (eg. v4.5) is a newer version than
   the writer of the file (eg. v4.3).
 - *Forward compatibility*, when the reader (eg. v4.5) is an older version than
   the writer of the file (eg. v4.7).

Backward compatibility is implemented by the reader when required, in this case
the reader simply does not access fields that were not present in the listed
version. Note that the offsets of fields may change across major versions, the
reader is responsible for implementing any differences.

Forward compatibility is implemented by the format specification and is only
available across minor versions. For readers, this means:
 - All fields supported by the reader will be accessible, but fields added
   later than the reader's supported version will not be accessible.
 - Readers must ignore or error unknown enumeration values. This will not affect
   the availability of any fields. The reader is responsible for synthesizing a
   fallback result from available data.
 - Readers must always use the saved structure size for "expandable" structures
   (described below) as the stride for arrays, rather than the size in the
   reader's supported version.

"Expandable" structures may increase in size over the course of minor versions,
the converse are "fixed" structures which do not. The status of any particular
structure is noted in the **END** row of the structure's table.

The format specification relies on the following restrictions to preserve
forward compatibility to the greatest extent possible:
 - Fields and enumeration values must never be removed, replaced, or change
   meaning in ways that would break older readers.
 - The presence or interpretation of fields must not depend on enumeration
   values.
 - Fields may be added in any previously uninterpreted region: in "gaps" between
   previous fields or at the end of the structure if it is expandable.
 - Enumeration values may be added in previously unallocated values.

A breakage of any of these restrictions requires a major version bump, adding
new fields or enumeration values requires a minor version bump.

* * *

`meta.db` version 4.0
===================================
`meta.db` is a binary file listing various metadata for the database, including:
  - Performance metrics for the metrics measured at application run-time,
  - Calling contexts for metric values listed in sibling `*.db` files, and
  - A human-readable description of the database's contents.

The `meta.db` file starts with the following header:

 Hex | Name         | Ver. | Section (see the [Common file structure])
 ---:| ------------ | ---- | ---------------------------------------------------
`00:`|                    || See [Common file structure]
`10:`|`{sz,p}General`  |4.0| [General Properties][GPsec]
`20:`|`{sz,p}IdNames`  |4.0| [Identifier Names][INsec]
`30:`|`{sz,p}Metrics`  |4.0| [Performance Metrics][PMsec]
`40:`|`{sz,p}Context`  |4.0| [Context Tree][CTsec]
`50:`|`{sz,p}Strings`  |4.0| Common String Table
`60:`|`{sz,p}Modules`  |4.0| [Load Modules][LMsec]
`70:`|`{sz,p}Files`    |4.0| [Source Files][SFsec]
`80:`|`{sz,p}Functions`|4.0| [Functions][Fnsec]
`90:`| **END**            || Extendable, see [Reader compatibility]

[GPsec]: #metadb-general-properties-section
[INsec]: #metadb-hierarchical-identifier-names-section
[PMsec]: #metadb-performance-metrics-section
[CTsec]: #metadb-context-tree-section
[LMsec]: #metadb-load-modules-section
[SFsec]: #metadb-source-files-section
[Fnsec]: #metadb-functions-section

The `meta.db` file ends with an 8-byte footer, reading `_meta.db` in ASCII.

Additional notes:
 - The Common String Table section has no particular interpretation, it is used
   as a section to store strings for the [Load Modules section][LMsec],
   the [Source Files section][SFsec], and the [Functions section][Fnsec].


`meta.db` General Properties section
-----------------------------------------
The General Properties section starts with the following header:

 Hex | Type | Name     | Ver. | Description (see the [Formats legend])
 ---:| ---- | -------- | ---- | -----------------------------------------------
`A 8`|| **ALIGNMENT**        || See [Alignment properties]
`00:`|char*|`pTitle`      |4.0| Title of the database. May be provided by the user.
`08:`|char*|`pDescription`|4.0| Human-readable Markdown description of the database.
`10:`|| **END**              || Extendable, see [Reader compatibility]

`description` provides information about the measured execution and subsequent
analysis that may be of interest to users. The exact layout and the information
contained may change without warning.

Additional notes:
 - The strings pointed to by `pTitle` and `pDescription` are fully contained
   within the General Properties section, including the terminating NUL byte.


`meta.db` Hierarchical Identifier Names section
-----------------------------------------------
> In future versions of HPCToolkit new functionality may be added that requires
> new values for the the `kind` field of a [Hierarchical Identifier Tuple][HIT].
> The Hierarchical Identifier Names section provides human-readable names for
> all possible values for forward compatibility.

The Hierarchical Identifier Names section starts with the following header:

 Hex | Type | Name   | Ver. | Description (see the [Formats legend])
 ---:| ---- | ------ | ---- | --------------------------------------------------
`A 8`|| **ALIGNMENT**      || See [Alignment properties]
`00:`|{Names}*|`ppNames`|4.0| Human-readable names for Identifier kinds
`08:`|u8|`nKinds`       |4.0| Number of names listed in this section
`09:`|| **END**            || Extendable, see [Reader compatibility]

{Names} above refers to a char*[`nKinds`] structure, ie. an array of `nKinds`
pointers to human-readable names for Identifier kinds. `ppNames[kind]` is the
human-readable name for the Identifier kind `kind`, where `kind` is part of a
[Hierarchical Identifier Tuple][HIT].

[HIT]: #profiledb-hierarchical-identifier-tuple-section

Additional notes:
 - The strings pointed to `ppNames[...]` are fully contained within the
   Hierarchical Identifier Names section, including the terminating NUL.


`meta.db` Performance Metrics section
-------------------------------------
> The Performance Metrics section lists the performance metrics measured at
> application runtime, and the analysis performed by HPCToolkit to generate the
> metric values within `profile.db` and `cct.db`. In summary:
>   - Performance measurements for an application thread are first attributed to
>     contexts, listed in the [Context Tree section](#metadb-context-tree-section).
>     These are the raw metric values.
>   - Propagated metric values are generated for each context by summing values
>     attributed to children contexts, within the measurements for a single
>     application thread. Which children are included in this sum is indicated
>     by the `*pScope` {PS} structure.
>   - Summary statistic values are generated for each context from the
>     propagated metric values for each application thread, by first applying
>     `*pFormula` to each value and then combining via `combine`. This generates
>     a single statistic value for each context.

The Performance Metrics section starts with the following header:

 Hex | Type | Name             | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---------------- | ---- | ----------------------------------------
`A 8`|| **ALIGNMENT**                || See [Alignment properties]
`00:`|{MD}[`nMetrics`]*|`pMetrics`|4.0| Descriptions of performance metrics
`08:`|u32|`nMetrics`              |4.0| Number of performance metrics
`0c:`|u8|`szMetric`               |4.0| Size of the {MD} structure, currently 32
`0d:`|u8|`szScopeInst`            |4.0| Size of the {PSI} structure, currently 16
`0e:`|u8|`szSummary`              |4.0| Size of the {SS} structure, currently 24
|    |
`10:`|{PS}[`nScopes`]*|`pScopes`  |4.0| Descriptions of propgation scopes
`18:`|u16|`nScopes`               |4.0| Number of propgation scopes
`1a:`|u8|`szScope`                |4.0| Size of the {PS} structure, currently 16
`1b:`|| **END**                      || Extendable, see [Reader compatibility]

{MD} above refers to the following sub-structure:

 Hex | Type | Name                    | Ver. | Description (see the [Formats legend])
 ---:| ---- | ----------------------- | ---- | -----------------------------------
`A 8`|| **ALIGNMENT**                       || See [Alignment properties]
`00:`|char*|`pName`                      |4.0| Canonical name for the metric
`08:`|{PSI}[`nScopeInsts`]*|`pScopeInsts`|4.0| Instantiated propagated sub-metrics
`10:`|{SS}[`nSummaries`]*|`pSummaries`   |4.0| Descriptions of generated summary statistics
`18:`|u16|`nScopeInsts`                  |4.0| Number of instantiated sub-metrics for this metric
`1a:`|u16|`nSummaries`                   |4.0| Number of summary statistics for this metric
|    |
`20:`|| **END**                             || Extendable, see [Reader compatibility]

{PSI} above refers to the following sub-structure:

 Hex | Type | Name      | Ver. | Description (see the [Formats legend])
 ---:| ---- | --------- | ---- | --------------------------------------------------
`A 8`|| **ALIGNMENT**         || See [Alignment properties]
`00:`|{PS}*|`pScope`       |4.0| Propagation scope instantiated
`08:`|u16|`propMetricId`   |4.0| Unique identifier for propagated metric values
|    |
`10:`|| **END**               || Extendable, see [Reader compatibility]

{SS} above refers to the following sub-structure:

 Hex | Type | Name   | Ver. | Description (see the [Formats legend])
 ---:| ---- | ------ | ---- | --------------------------------------------------
`A 8`|| **ALIGNMENT**      || See [Alignment properties]
`00:`|{PS}*|`pScope`    |4.0| Propagation scope summarized
`08:`|char*|`pFormula`  |4.0| Canonical unary function used for summary values
`10:`|u8|`combine`      |4.0| Combination n-ary function used for summary values
|    |
`12:`|u16|`statMetricId`|4.0| Unique identifier for summary statistic values
|    |
`18:`|| **END**            || Extendable, see [Reader compatibility]

The combination function `combine` is an enumeration with the following possible
values (the name after `/` is the matching name for `inputs:combine` in
METRICS.yaml):
 - `0/sum`: Sum of input values
 - `1/min`: Minimum of input values
 - `2/max`: Maximum of input values

> As noted before, propagated metric values are generated by summing the
> measured metric values from some or all of the descendants of a context, as
> determined by the "propagation scope." The fields of the {PS} structure
> describe the function mapping from each context to the subset of its
> descendants included in the sum. While presentable metric values can be
> produced based only on the `*pScopeName` and METRICS.yaml, more complex
> analysis can be performed if the propagation scope is defined in the meta.db.

{PS} above refers to the following sub-structure:

 Hex | Type | Name      | Ver. | Description (see the [Formats legend])
 ---:| ---- | --------- | ---- | -----------------------------------------------
`A 8`|| **ALIGNMENT**         || See [Alignment properties]
`00:`|char*|`pScopeName`   |4.0| Name of the propagation scope
`08:`|u8|`type`            |4.0| Type of propagation scope described
`09:`|u8|`propagationIndex`|4.0| Index of this propagation's propagation bit
|    |
`10:`|| **END**              || Extendable, see [Reader compatibility]

> The propagation scope's name `*pScopeName` may be any string, however to aid
> writing and maintaining metric taxonomies (see METRICS.yaml) the propagation
> scope for a name rarely changes meanings. Regardless, readers are still
> strongly encouraged to use the definition provided by other fields described
> below to perform analysis requiring any non-trivial understanding of the
> propagation scope.

The propagation scope `type` is an enumeration with the following values:
 - `0`: Custom propagation scope, not defined in the meta.db.

 - `1`: Standard "point" propagation scope. No propagation occurs, all metric
   values are recorded as measured.

   > The canonical `*pScopeName` for this case is "point".

 - `2`: Standard "execution" propagation scope. Propagation always occurs, the
   propagated value sums values measured from all descendants without exception.

   > The canonical `*pScopeName` for this case is "execution". This case is
   > often used for inclusive metric costs.

 - `3`: Transitive propagation scope. Propagation occurs from every context to
   its parent when the `propagationIndex`th bit is set in the
   [context's `propagation` bitmask][CT], and to further ancestors transitively
   under the same condition.

   > An example transitive propagation scope is named "function," its propagated
   > values sum measurements from all descendants not separated by a call, in
   > other words its the metric cost exclusive to the function. The appropriate
   > bit in the `propagation` bitmask is set only if the context's `relation`
   > is a caller-callee relationship (of some kind).

[CT]: #metadb-context-tree-section

Additional notes:
 - Matching the size of the [context's `propagation` field][CT],
   `propagationIndex` is always less than 16.
 - The arrays pointed to by `pMetrics`, `pScopes` and `pSummaries` are fully
   contained within the Performance Metrics section.
 - The strings pointed to by `pName`, `pScope` and `pFormula` are fully contained
   within the Performance Metrics section, including the terminating NUL.
 - The format and interpretation of `*pFormula` matches the `inputs:formula` key
   in METRICS.yaml, see there for details.
 - `propMetricId` is the metric identifier used in `profile.db` and `cct.db` for
   propagated metric values for the given `*pName` and `*pScope`.
 - `statMetricId` is the metric identifier used in `profile.db` summary
   profiles for summary statistic values for the given `*pName`, `*pScope`,
   `*pFormula` and `combine`.
 - The stride of `*pMetrics`, `*pScopes` and `*pSummaries` is `szMetric`,
   `szScope` and `szSummary`, respectively. For forward compatibility these
   values should be read and used whenever accessing these arrays.

`meta.db` Load Modules section
-----------------------------
> The Load Modules section lists information about the binaries used during the
> measured application execution.

The Load Modules section starts with the following header:

 Hex | Type | Name | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---- | ---- | ----------------------------------------------------
`A 8`|| **ALIGNMENT**    || See [Alignment properties]
`00:`|[LMS]\[`nModules`]*|`pModules`|4.0| Load modules used in this database
`08:`|u32|`nModules`  |4.0| Number of load modules listed in this section
`0c:`|u16|`szModule`  |4.0| Size of a [Load Module Specification][LMS], currently 16
`0e:`|| **END**          || Extendable, see [Reader compatibility]

[LMS]: #load-module-specification

Additional notes:
 - The array pointed to by `pModules` is completely within the Load Modules
   section.
 - The stride of `*pModules` is `szModule`, for forwards compatibility this
   should always be read and used as the stride when accessing `*pModules`.

### Load Module Specification ###
A Load Module Specification refers to the following structure:

 Hex | Type | Name | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---- | ---- | ----------------------------------------------------
`A 8`|| **ALIGNMENT**    || See [Alignment properties]
`00:`|u32|`flags`     |4.0| Reserved for future use
|    |
`08:`|char*|`pPath`   |4.0| Full path to the associated application binary
`10:`|| **END**          || Extendable, see [Reader compatibility]

Additional notes:
 - The string pointed to by `pPath` is completely within the
   [Common String Table section](#metadb-version-40), including the terminating
   NUL byte.


`meta.db` Source Files section
-----------------------------
> The Source Files section lists information about the application's source
> files, as gathered through debugging information on application binaries.

The Source Files section starts with the following header:

 Hex | Type | Name | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---- | ---- | ----------------------------------------------------
`A 8`|| **ALIGNMENT**    || See [Alignment properties]
`00:`|[SFS]\[`nFiles`]*|`pFiles`|4.0| Source files used in this database
`08:`|u32|`nFiles`    |4.0| Number of source files listed in this section
`0c:`|u16|`szFile`    |4.0| Size of a [Source File Specification][SFS], currently 16
`0e:`|| **END**          || Extendable, see [Reader compatibility]

[SFS]: #source-file-specification

Additional notes:
 - The array pointed to by `pFiles` is completely within the Source Files
   section.
 - The stride of `*pFiles` is `szFile`, for forwards compatibility this should
   always be read and used as the stride when accessing `*pFiles`.

### Source File Specification ###
A Source File Specification refers to the following structure:

 Hex | Type | Name | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---- | ---- | ----------------------------------------------------
`A 8`|| **ALIGNMENT**    || See [Alignment properties]
`00:`|{Flags}|`flags` |4.0| See below
|    |
`08:`|char*|`pPath`   |4.0| Path to the source file. Absolute, or relative to the root database directory.
`10:`|| **END**          || Extendable, see [Reader compatibility]

{Flags} refers to an u32 bitfield with the following sub-fields (bit 0 is
least significant):
 - Bit 0: `copied`. If 1, the source file was copied into the database and
   should always be available. If 0, the source file was not copied and thus may
   need to be searched for.
 - Bits 1-31: Reserved for future use.

Additional notes:
 - The string pointed to by `pPath` is completely within the
  [Common String Table section](#metadb-version-40), including the terminating
   NUL byte.


`meta.db` Functions section
------------------------------
> The Functions section lists various named source-level constructs observed in
> the application. These are inclusively called "functions," however this also
> includes other named constructs (e.g. `<gpu kernel>`).
>
> Counter-intuitively, sometimes we know a named source-level construct should
> exist, but not its actual name. These are "anonymous functions," in this case
> it is the reader's responsibility to construct a reasonable name with what
> information is available.

The Functions section starts with the following header:

 Hex | Type | Name   | Ver. | Description (see the [Formats legend])
 ---:| ---- | ------ | ---- | --------------------------------------------------
`A 8`|| **ALIGNMENT**      || See [Alignment properties]
`00:`|[FS]\[`nFunctions`]*|`pFunctions`|4.0| Functions used in this database
`08:`|u32|`nFunctions`  |4.0| Number of functions listed in this section
`0c:`|u16|`szFunction`  |4.0| Size of a [Function Specification][FS], currently 40
`0e:`|| **END**            || Extendable, see [Reader compatibility]

[FS]: #function-specification

Additional notes:
 - The array pointed to by `pFunctions` is completely within the Functions
   section.
 - The stride of `*pFunctions` is `szFunction`, for forwards compatibility this
   should always be read and used as the stride when accessing `*pFunctions`.

### Function Specification ###
A Function Specification refers to the following structure:

 Hex | Type | Name | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---- | ---- | ----------------------------------------------------
`A 8`|| **ALIGNMENT**    || See [Alignment properties]
`00:`|char*|`pName`   |4.0| Human-readable name of the function, or 0
`08:`|[LMS]*|`pModule`|4.0| Load module containing this function, or 0
`10:`|u64|`offset`    |4.0| Offset within `*pModule` of this function's entry point
`18:`|[SFS]*|`pFile`  |4.0| Source file of the function's definition, or 0
`20:`|u32|`line`      |4.0| Source line in `*pFile` of the function's definition
`24:`|u32|`flags`     |4.0| Reserved for future use
`28:`|| **END**          || Extendable, see [Reader compatibility]

[LMS]: #load-module-specification
[SFS]: #source-file-specification


Additional notes:
 - If not 0, the string pointed to by `pName` is completely within the
   [Common String Table section](#metadb-version-40), including the terminating
   NUL byte.
 - If not 0, `pModule` points within the [Load Module section](#metadb-load-module-section).
 - If not 0, `pFile` points within the [Source File section](#metadb-source-file-section).
 - At least one of `pName`, `pModule` and `pFile` will not be 0.


`meta.db` Context Tree section
-------------------------------
> The Context Tree section lists the source-level calling contexts in which
> performance data was gathered during application runtime. Each context
> ({Ctx} below) represents a source-level (lexical) context, and these can be
> nested to create paths in the tree:
>
>     function foo()
>       loop at foo.c:12
>         line foo.c:15
>           instruction libfoo.so@0x10123
>
> The relation between contexts may be enclosing lexical context (as above), or
> can be marked as a call of various types (see `relation` below):
>
>     instruction libfoo.so@0x10123
>       [normal call to] function bar()
>         line bar.c:57
>           [inlined call to] function baz()
>             instruction libbar.so@0x25120
>
> Although some patterns in the context tree are more common than others, the
> format is very flexible and will allow almost any nested structure. It is up
> to the reader to interpret the context tree as appropriate.
>
> The top of the context tree is made of "entry points" that describe how the
> application's code was called from the operating system.

The Context Tree section starts with the following header:

 Hex | Type | Name     | Ver. | Description (see the [Formats legend])
 ---:| ---- | -------- | ---- | ------------------------------------------------
`A 8`|| **ALIGNMENT**        || See [Alignment properties]
`00:`|{Entry}[`nEntryPoints`]*|`pEntryPoints`|4.0| Pointer to an array of entry point specifications
`08:`|u16|`nEntryPoints`  |4.0| Number of entry points in this context tree
`0a:`|u8|`szEntryPoint`   |4.0| Size of a {Entry} structure in bytes, currently 32
`0b:`|| **END**              || Extendable, see [Reader compatibility]

{Entry} above refers to the following structure:

 Hex | Type | Name        | Ver. | Description (see the [Formats legend])
 ---:| ---- | ----------- | ---- | ---------------------------------------------
`A 8`|| **ALIGNMENT**           || See [Alignment properties]
`00:`|u64|`szChildren`       |4.0| Total size of `*pChildren`, in bytes
`08:`|{Ctx}[...]*|`pChildren`|4.0| Pointer to the array of child contexts
`10:`|u32|`ctxId`            |4.0| Unique identifier for this context
`14:`|u16|`entryPoint`       |4.0| Type of entry point used here
|    |
`18:`|char*|`pPrettyName`    |4.0| Human-readable name for the entry point
`20:`|| **END**              || Extendable, see [Reader compatibility]

`entryPoint` is an enumeration of the following possible values (name in quotes
is the associated canonical `*pPrettyName`):
 - `0` "unknown entry": No recognized outside caller.

   > This can occur when the unwind fails due to incomplete unwind information.
 - `1` "main thread": Setup code for the main thread.
 - `2` "application thread": Setup code for threads created by the application,
   via `pthread_create` or similar.

Additional notes:
 - The string pointed to by `pPrettyName` is completely within the
   [Common String Table section](#metadb-version-40), including the terminating
   NUL byte.

{Ctx} above refers to the following structure:

 Hex | Type | Name            | Ver. | Description (see the [Formats legend])
 ---:| ---- | --------------- | ---- | -----------------------------------------
`A 8`|| **ALIGNMENT**               || See [Alignment properties]
`00:`|u64|`szChildren`           |4.0| Total size of `*pChildren`, in bytes
`08:`|{Ctx}[...]*|`pChildren`    |4.0| Pointer to the array of child contexts
`10:`|u32|`ctxId`                |4.0| Unique identifier for this context
`14:`|{Flags}|`flags`            |4.0| See below
`15:`|u8|`relation`              |4.0| Relation this context has with its parent
`16:`|u8|`lexicalType`           |4.0| Type of lexical context represented
`17:`|u8|`nFlexWords`            |4.0| Size of `flex`, in u8[8] "words" (bytes / 8)
`18:`|u16|`propagation`          |4.0| Bitmask for defining [propagation scopes][PMS]
|    |
`20:`|u8[8]\[`nFlexWords`]|`flex`|4.0| Flexible data region, see below
` *:`|| **END**

[PMS]: #metadb-performance-metrics-section

`flex` contains a dynamic sequence of sub-fields, which are sequentially
"packed" into the next unused bytes at the minimum alignment. In particular:
 - An u64 sub-field will always take the next full u8[8] "word" and never span
   two words, but
 - Two u32 sub-fields will share a single u8[8] word even if an u64 sub-field
   is between them in the packing order.

The packing order is indicated by the index on `flex`, ie. `flex[1]` is the
sub-field next in the packing order after `flex[0]`. This order still holds
even if not all fields are present for any particular instance.

{Flags} above refers to an u8 bitfield with the following sub-fields (bit 0 is
least significant):
 - Bit 0: `hasFunction`. If 1, the following sub-fields of `flex` are present:
   + `flex[0]:` [FS]* `pFunction`: Function associated with this context
 - Bit 1: `hasSrcLoc`. If 1, the following sub-fields of `flex` are present:
   + `flex[1]:` [SFS]* `pFile`: Source file associated with this context
   + `flex[2]:` u32 `line`: Associated source line in `pFile`
 - Bit 2: `hasPoint`. If 1, the following sub-fields of `flex` are present:
   + `flex[3]:` [LMS]* `pModule`: Load module associated with this context
   + `flex[4]:` u64 `offset`: Associated byte offset in `*pModule`
 - Bits 3-7: Reserved for future use.

[FS]: #function-specification
[SFS]: #source-file-specification
[LMS]: #load-module-specification

`relation` is an enumeration with the following values:
 - `0`: This context's parent is an enclosing lexical context, eg. source line
   within a function. Specifically, no call occurred.
 - `1`: This context's parent used a typical function call to reach this
   context. The parent context is the source-level location of the call.
 - `2`: This context's parent used an inlined function call (ie. the call was
   inlined by the compiler). The parent context is the source-level location of
   the original call.

The lexical type `lexicalType` is an enumeration with the following values:
 - `0`: Function-like construct. If `hasFunction` is 1, `*pFunction` indicates
   the function represented by this context. Otherwise the function for this
   context is unknown (ie. an unknown function).
 - `1`: Loop construct. `*pFile` and `line` indicate the source line of the
   loop header.
 - `2`: Source line construct. `*pFile` and `line` indicate the source line
   represented by this context.
 - `3`: Single instruction. `*pModule` and `offset` indicate the first byte of the
   instruction represented by this context.

Additional notes:
 - `propagation` is an extra field used to assist in defining some propagation
   scopes, see the [Performance Metrics section][PMS] for details.
 - `ctxId` is always larger than 0, the value 0 is reserved to indicate the
   global context (ie. an implicit context above and enclosing all others).

   > The global context is used to represent corner cases where there should be
   > no associated context. See notes on the usage of `ctxId` for details.
 - The arrays pointed to by `pRoots` and `pChildren` are completely within the
   Context Tree section. The size of these arrays is given in `szRoots` or
   `szChildren`, in bytes to allow for a singular read of all root/child
   context structures.
 - `pChildren` is 0 if there are no child Contexts, `pRoots` is 0 if there are
   no Contexts in this section period. `szChildren` and `szRoots` are 0 in these
   cases respectively.
 - `pFunction` points within the [Function section](#metadb-function-section).
 - `pFile` points within the [Source File section](#metadb-source-file-section).
 - `pModule` points within the [Load Module section](#metadb-load-module-section).
 - The size of a single {Ctx} is dynamic but can be derived from `nFlexWords`.
   For forward compatibility, readers should always read and use this to read
   arrays of {Ctx} elements.


* * *
`profile.db` version 4.0
========================

The `profile.db` is a binary file containing the performance analysis results
generated by `hpcprof`, arranged by application thread and in a compact sparse
representation. Once an application thread is chosen, the analysis result for
a particular calling context can be obtained through a simple binary search.

The `profile.db` file starts with the following header:

 Hex | Name            | Ver. | Section (see the [Common file structure])
 ---:| --------------- | ---- | ------------------------------------------------
`00:`|                       || See [Common file structure]
`10:`|`{sz,p}ProfileInfos`|4.0| [Profiles Information][PIsec]
`20:`|`{sz,p}IdTuples`    |4.0| [Hierarchical Identifier Tuples][HITsec]
`30:`| **END**               || Extendable, see [Reader compatibility]

[PIsec]: #profiledb-profile-info-section
[HITsec]: #profiledb-hierarchical-identifier-tuple-section

The `profile.db` file ends with an 8-byte footer, reading `_prof.db` in ASCII.


`profile.db` Profile Info section
---------------------------------
> The Profile Info section lists the CPU threads and GPU streams present in the
> application execution, and references the performance *profile* measured from
> it during application runtime. Profiles may contain summary statistic values
> or propagated metric values, the first always contains summary statistic
> values from the entire application execution, intended to aid top-down
> analysis. See the comment in the [`meta.db` Performance Metrics section](#metadb-performance-metrics-section)
> for more detail on how values are calculated.

The Profile Info section starts with the following header:

 Hex | Type | Name  | Ver. | Description (see the [Formats legend])
 ---:| ---- | ----- | ---- | ---------------------------------------------------
`A 8`|| **ALIGNMENT**     || See [Alignment properties]
`00:`|{PI}[`nProfiles`]*|`pProfiles`|4.0| Description for each profile
`08:`|u32|`nProfiles`  |4.0| Number of profiles listed in this section
`0c:`|u8|`szProfile`   |4.0| Size of a {PI} structure, currently 40
`0d:`|| **END**           || Extendable, see [Reader compatibility]

{PI} above refers to the following structure:

 Hex | Type | Name    | Ver. | Description (see the [Formats legend])
 ---:| ---- | ------- | ---- | -------------------------------------------------
`A 8`|| **ALIGNMENT**       || See [Alignment properties]
`00:`|[PSVB]|`valueBlock`|4.0| Header for the values for this profile
`20:`|[HIT]*|`pIdTuple`  |4.0| Identifier tuple for this profile
`28:`|{Flags}|`flags`    |4.0| See below
|    |
`30:`|| **END**             || Extendable, see [Reader compatibility]

[HIT]: #profiledb-hierarchical-identifier-tuple-section
[PSVB]: #profile-major-sparse-value-block

{Flags} above refers to a u32 bitfield with the following sub-fields (bit 0
is least significant):
 - Bit 0: `isSummary`. If 0, this profile is a performance profile of the
   application thread identified exactly by `*pIdTuple`. If 1, this profile is a
   "summary profile" containing statistics across multiple measured application
   threads where `*pIdTuple` lists common identifiers.

Additional notes:
 - The array pointed to by `pProfiles` is fully contained within the Profile
   Info section.
 - Profiles are unordered within this section, except the first which is
   always the "canonical summary profile." This is always a summary profile and
   contains statistics across all measured application threads.
 - `pIdTuple` points within the [Identifier Tuple section](#profledb-hierarchical-identifier-tuple-section),
   except for the canonical summary profile where `pIdTuple` is 0.
 - The stride of `*pProfiles` is equal to `szProfile`, for forward compatibility
   this should always be read and used as the stride when accessing `*pProfiles`.

### Profile-Major Sparse Value Block ###
> All performance data in the `profile.db` and `cct.db` is arranged in
> internally-sparse "blocks," the variant present in `profile.db` uses one for
> each application thread (or equiv.) measured during application runtime.
> Conceptually, these are individual planes of a 3-dimensional tensor indexed by
> application thread (profile), context, and metric, in that order.

Each Profile-Major Sparse Value Block has the following structure:

 Hex | Type | Name              | Ver. | Description (see the [Formats legend])
 ---:| ---- | ----------------- | ---- | ---------------------------------------
`A 8`|| **ALIGNMENT**                 || See [Alignment properties]
`00:`|u64|`nValues`                |4.0| Number of non-zero values
`08:`|{Val}[`nValues`]*|`pValues`  |4.0| Metric-value pairs
`10:`|u32|`nCtxs`                  |4.0| Number of non-empty contexts
|    |
`18:`|{Idx}[`nCtxs`]*|`pCtxIndices`|4.0| Mapping from contexts to values
`20:`|| **END**                       || Fixed, see [Reader compatibility]

{Val} above refers to the following structure:

 Hex | Type | Name | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---- | ---- | ----------------------------------------------------
`A 2`|| **ALIGNMENT**    || See [Alignment properties]
`00:`|u16|`metricId`  |4.0| Unique identifier of a metric listed in the [`meta.db`](#metadb-performance-metrics-section)
`02:`|f64|`value`     |4.0| Value of the metric indicated by `metricId`
`0a:`|| **END**          || Fixed, see [Reader compatibility]

{Idx} above refers to the following structure:

 Hex | Type | Name | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---- | ---- | ----------------------------------------------------
`A 4`|| **ALIGNMENT**    || See [Alignment properties]
`00:`|u32|`ctxId`     |4.0| Unique identifier of a context listed in the [`meta.db`](#metadb-context-tree-section)
`04:`|u64|`startIndex`|4.0| Start index of `*pValues` attributed to the referenced context
`0c:`|| **END**          || Fixed, see [Reader compatibility]

The sub-array of `*pValues` attributed to the context referenced by `ctxId`
starts at index `startIndex` and ends just before the `startIndex` of the
following {Idx} structure, if this {Idx} is the final element of `*pCtxIndices`
(index `nCtxs - 1`) then the end is the last element of `*pValues` (index
`nValues - 1`).

Additional notes:
 - `pValues` and `pCtxIndices` point outside the sections listed in the
   [`profile.db` header](#profiledb-version-40).
 - The arrays pointed to by `pValues` and `pCtxIndices` are subsequent: only
   padding is placed between them and `pValues < pCtxIndices`. This allows
   readers to read a plane of data in a single contiguous blob from `pValues`
   to `pCtxIndices + nCtxs * 0xc`.
 - `metricId` is a `propMetricId` or `statMetricId` listed in the
   [`meta.db` performance metrics section](#metadb-performance-metrics-section),
   this is a `statMetricId` if `isSummary` is 1 and a `propMetricId` otherwise.
 - `ctxId` is a `ctxId` listed in the [`meta.db` context tree section](#metadb-context-tree-section),
   or 0 for metric values attributed to the implicit global context.

   > The values attributed to the global context are propagated from the root
   > contexts, in effect these values give an "aggregate" view of the profile
   > where the context dimension has been removed.
 - `*pValues` and `*pCtxIndices` are sorted by `metricId` and `ctxId`,
   respectively. This allows the use of binary search (or some variant thereof)
   to locate the value(s) for a particular context or metric.
 - `value` and `startIndex` are not aligned, however `metricId` and `ctxId` are.
   This should in general not pose a significant performance penalty.
   See [Alignment properties] above.


`profile.db` Hierarchical Identifier Tuple section
--------------------------------------------------
> Application threads (or equiv.) are identified in a human-readable manner via
> the use of Hierarchical Identifier Tuples. Each tuple lists a series of
> identifications for an application thread, for instance compute node and
> MPI rank. The identifications within a tuple are ordered roughly by
> "hierarchy" from largest to smallest, for eg. compute node will appears before
> MPI rank if both are present.

The Hierarchical Identifier Tuple section contains multiple Identifier Tuples,
each of the following structure:

 Hex | Type | Name   | Ver. | Description (see the [Formats legend])
 ---:| ---- | ------ | ---- | --------------------------------------------------
`A 8`|| **ALIGNMENT**      || See [Alignment properties]
`00:`|u16|`nIds`        |4.0| Number of identifications in this tuple
|    |
`08:`|{Id}[`nIds`]|`ids`|4.0| Identifications for an application thread
` *:`|| **END**            || Extendable, see [Reader compatibility]

{Id} above refers to the following structure:

 Hex | Type | Name    | Ver. | Description (see the [Formats legend])
 ---:| ---- | ------- | ---- | -------------------------------------------------
`A 8`|| **ALIGNMENT**       || See [Alignment properties]
`00:`|u8|`kind`          |4.0| One of the values listed in the [`meta.db` Identifier Names section][INsec].
|    |
`02:`|{Flags}|`flags`    |4.0| See below.
`04:`|u32|`logicalId`    |4.0| Logical identifier value, may be arbitrary but dense towards 0.
`08:`|u64|`physicalId`   |4.0| Physical identifier value, eg. hostid or PCI bus index.
`10:`|| **END**             || Fixed, see [Reader compatibility]

[INsec]: #metadb-hierarchical-identifier-names-section

{Flags} above refers to an u16 bitfield with the following sub-fields (bit 0 is
least significant):
 - Bit 0: `isPhysical`. If 1, the `kind` represents a physical (hardware or VM)
   construct for which `physicalId` is the identifier (and `logicalId` is
   arbitrary but distinct). If 0, `kind` represents a logical (software-only)
   construct (and `physicalId` is `logicalId` zero-extended to 64 bits).
 - Bits 1-15: Reserved for future use.

> The name associated with the `kind` in the [`meta.db`][INsec] indicates the
> meaning of `logicalId` (if `isPhysical == 0`) and/or `physicalId` (if
> `isPhysical == 1`). The following names are in current use with the given
> meanings:
>  - "NODE": Compute node, `physicalId` indicates the hostid of the node.
>  - "RANK": Rank of the process (from eg. MPI), `logicalId` indicates the rank.
>  - "CORE": Core the application thread was bound to, `physicalId` indicates
>    the index of the first hardware thread as listed in /proc/cpuinfo.
>  - "THREAD": Application CPU thread, `logicalId` indicates the index.
>  - "GPUCONTEXT": Context used to access a GPU, `logicalId` indicates the index
>    as given by the underlying programming model (eg. CUDA context index).
>  - "GPUSTREAM": Stream/queue used to push work to a GPU, `logicalId` indicates
>    the index as given by the programming model (eg. CUDA stream index).
>
> These names/meanings are not stable and may change without a version bump, it
> is highly recommended that readers refrain from any special-case handling of
> particular `kind` values where possible.

Additional notes:
 - While `physicalId` (when valid) lists a physical identification for an
   application thread, the contained value is often too obtuse for generating
   human-readable output listing many identifiers. `logicalId` is a suitable
   replacement in these cases, as these values are always dense towards 0.


* * *
`cct.db` version 4.0
====================

The `cct.db` is a binary file containing the performance analysis results
generated by `hpcprof`, arranged by calling context and in a compact sparse
representation. Once a calling context is chosen, the analysis result for
a particular metric can be obtained through a simple binary search.

The `cct.db` file starts with the following header:

 Hex | Name        | Ver. | Section (see the [Common file structure])
 ---:| ----------- | ---- | ----------------------------------------------------
`00:`|                   || See [Common file structure]
`10:`|`{sz,p}CtxInfos`|4.0| [Contexts Information][CIsec]
`20:`| **END**           || Extendable, see [Reader compatibility]

[CIsec]: #cctdb-context-info-section

The `cct.db` file ends with an 8-byte footer, reading `__ctx.db` in ASCII.


`cct.db` Context Info section
-----------------------------
> The Context Info section associates contexts with a "block" of performance
> data, similar to what the [`profile.db` Profile Info section](#metadb-profiledb-profile-info-section)
> does for application threads.

The Context Info section starts with the following header:

 Hex | Type | Name       | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---------- | ---- | ----------------------------------------------
`A 8`|| **ALIGNMENT**          || See [Alignment properties]
`00:`|{CI}[`nCtxs`]*|`pCtxs`|4.0| Description for each context in this database
`08:`|u32|`nCtxs`           |4.0| Number of contexts listed in this section
`0c:`|u8|`szCtx`            |4.0| Size of a {CI} structure, currently 32
`0d:`|| **END**                || Extendable, see [Reader compatibility]

{CI} above refers to the following structure:

 Hex | Type | Name    | Ver. | Description (see the [Formats legend])
 ---:| ---- | ------  | ---- | -------------------------------------------------
`A 8`|| **ALIGNMENT**       || See [Alignment properties]
`00:`|[CSVB]|`valueBlock`|4.0| Header for the values for this Context
`20:`|| **END**             || Extendable, see [Reader compatibility]

[CSVB]: #context-major-sparse-value-block

Additional notes:
 - The array pointed to by `pCtxs` is fully contained within the Context Info
   section.
 - `(*pCtxs)[ctxId]` is associated with the context with the matching `ctxId`
   as listed in the [`meta.db` context tree section](#metadb-context-tree-section).
   `(*pCtxs)[0]` contains the metric values of the implicit global context.

   > The values attributed to the global context are propagated from the root
   > contexts, in effect these values give an "aggregate" view of the profile
   > where the context dimension has been removed.
 - The stride of `*pCtxs` is `szCtx`, for forward compatibility this should
   always be read and used as the stride when accessing `*pCtxs`.

### Context-Major Sparse Value Block ###
> The Context-Major Sparse Value Block is very similar in structure to the
> [Profile-Major Sparse Value Block](#profile-major-sparse-value-block), they
> differ mainly in the order in which their 3 dimensions are indexed.
> For Context-Major, this order is context, metric, application thread (profile).

Each Context-Major Sparse Value Block has the following structure:

 Hex | Type | Name              | Ver. | Description (see the [Formats legend])
 ---:| ---- | ----------------- | ---- | ---------------------------------------
`A 8`|| **ALIGNMENT**                 || See [Alignment properties]
`00:`|u64|`nValues`                |4.0| Number of non-zero values
`08:`|{Val}[`nValues`]*|`pValues`  |4.0| Profile-value pairs
`10:`|u16|`nMetrics`               |4.0| Number of non-empty metrics
|    |
`18:`|{Idx}[`nMetrics`]*|`pMetricIndices`|4.0| Mapping from metrics to values
`20:`|| **END**                       || Fixed, see [Reader compatibility]

{Val} above refers to the following structure:

 Hex | Type | Name | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---- | ---- | ----------------------------------------------------
`A 4`|| **ALIGNMENT**    || See [Alignment properties]
`00:`|u32|`profIndex` |4.0| Index of a profile listed in the [`profile.db`](#profiledb-profile-info-section)
`04:`|f64|`value`     |4.0| Value attributed to the profile indicated by `profIndex`
`0c:`|| **END**          || Fixed, see [Reader compatibility]

{Idx} above refers to the following structure:

 Hex | Type | Name | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---- | ---- | ----------------------------------------------------
`A 2`|| **ALIGNMENT**    || See [Alignment properties]
`00:`|u16|`metricId`  |4.0| Unique identifier of a metric listed in the [`meta.db`](#metadb-performance-metrics-section)
`02:`|u64|`startIndex`|4.0| Start index of `*pValues` from the associated metric
`0a:`|| **END**          || Fixed, see [Reader compatibility]

The sub-array of `*pValues` from to the metric referenced by `metId` starts at
index `startIndex` and ends just before the `startIndex` of the following {Idx}
structure, if this {Idx} is the final element of `*pCtxIndices` (index
`nMetrics - 1`) then the end is the last element of `*pValues` (index
`nValues - 1`).

Additional notes:
 - `pValues` and `pMetricIndices` point outside the sections listed in the
   [`cct.db` header](#cctdb-version-40).
 - The arrays pointed to by `pValues` and `pMetricIndices` are subsequent: only
   padding is placed between them and `pValues < pMetricIndices`. This allows
   readers to read a plane of data in a single contiguous blob from `pValues`
   to `pMetricIndices + nMetrics * 0xa`.
 - `metricId` is a `propMetricId` listed in the
   [`meta.db` performance metrics section](#metadb-performance-metrics-section).
   Unlike `profile.db`, the `cct.db` does not include any summary profiles.
 - `*pValues` and `*pMetricIndices` are sorted by `profIdx` and `metricId`,
   respectively. This allows the use of binary search (or some variant thereof)
   to locate the value(s) for a particular metric or application thread.
 - `value` and `startIndex` are not aligned, however `profIdx` and `metricId`
   are. This should in general not pose a significant performance penalty.
   See [Alignment properties] above.


* * *
`trace.db` version 4.0
======================

The `trace.db` file starts with the following header:

 Hex | Name         | Ver. | Section (see the [Common file structure])
 ---:| ------------ | ---- | ---------------------------------------------------
`00:`|                    || See [Common file structure]
`10:`|`{sz,p}CtxTraces`|4.0| [Context Trace Headers][CTHsec]
`20:`| **END**            || Extendable, see [Reader compatibility]

[CTHsec]: #tracedb-context-trace-headers-section

The `trace.db` file ends with an 8-byte footer, reading `trace.db` in ASCII.

`trace.db` Context Trace Headers section
--------------------------------

The Context Trace Headers sections starts with the following structure:

 Hex | Type | Name            | Ver. | Description (see the [Formats legend])
 ---:| ---- | --------------- | ---- | -----------------------------------------
`A 8`|| **ALIGNMENT**               || See [Alignment properties]
`00:`|{CTH}[`nTraces`]*|`pTraces`|4.0| Header for each trace
`08:`|u32|`nTraces`              |4.0| Number of traces listed in this section
`0c:`|u8|`szTrace`               |4.0| Size of a {TH} structure, currently 24
|    |
`10:`|u64|`minTimestamp`  |4.0| Smallest timestamp of the traces listed in `*pTraces`
`18:`|u64|`maxTimestamp`  |4.0| Largest timestamp of the traces listed in `*pTraces`
`20:`|| **END**              || Extendable, see [Reader compatibility]

{CTH} above refers to the following structure:

 Hex | Type | Name  | Ver. | Description (see the [Formats legend])
 ---:| ---- | ----- | ---- | ---------------------------------------------------
`A 8`|| **ALIGNMENT**     || See [Alignment properties]
`00:`|u32|`profIndex`  |4.0| Index of a profile listed in the [`profile.db`](#profiledb-profile-info-section)
|    |
`08:`|{Elem}*|`pStart` |4.0| Pointer to the first element of the trace line (array)
`10:`|{Elem}*|`pEnd`   |4.0| Pointer to the after-end element of the trace line (array)
`18:`|| **END**          || Extendable, see [Reader compatibility]

{Elem} above refers to the following structure:

 Hex | Type | Name | Ver. | Description (see the [Formats legend])
 ---:| ---- | ---- | ---- | ----------------------------------------------------
`A 4`|| **ALIGNMENT**    || See [Alignment properties]
`00:`|u64|`timestamp` |4.0| Timestamp of the trace sample (nanoseconds since the epoch)
`08:`|u32|`ctxId`     |4.0| Unique identifier of a context listed in [`meta.db`](#metadb-context-tree-section)
`0c:`|| **END**          || Fixed, see [Reader compatibility]

Additional notes:
 - If `ctxId` is 0, the traced thread was not running at the `timestamp`.
   Consecutive {Elem} elements cannot both have `ctxId` set to 0.

   > This is equivalent to attributing the trace element to the implicit global
   > context.
 - The array pointed to by `pTraces` is completely within the Context Trace
   Headers section. The pointers `pStart` and `pEnd` point outside any of the
   sections listed in the [`trace.db` header](#tracedb-version-40).
 - The array starting at `pStart` and ending just before `pEnd` is sorted in
   order of increasing `timestamp`.
 - The stride of `*pTraces` is `szTrace`, for forward compatibility this value
   should be read and used when accessing `*pTraces`.
 - `timestamp` is only aligned for even elements in a trace line array. Where
   possible, readers are encouraged to prefer accessing even elements.
   See [Alignment properties] above.
