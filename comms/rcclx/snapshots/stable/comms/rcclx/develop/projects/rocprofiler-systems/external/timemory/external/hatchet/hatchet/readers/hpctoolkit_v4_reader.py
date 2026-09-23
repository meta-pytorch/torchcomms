# Copyright 2021-2023 University of Maryland and other Hatchet Project
# Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import hatchet.graphframe
from hatchet.node import Node
from hatchet.graph import Graph
from hatchet.graphframe import GraphFrame
from hatchet.frame import Frame
import struct
import yaml


class MetaReader:
    # adds new context id and return new nid
    def _add_context_id(self, context_id) -> int:
        self.nid_to_ctx[self.current_nid] = context_id
        self.current_nid += 1
        return self.current_nid - 1

    def __init__(self, file_location):
        # open the file to ready in binary mode (rb)
        self.file = open(file_location, "rb")

        # setting necessary read options
        self.byte_order = "little"
        self.signed = False
        self.encoding = "ASCII"
        self.current_nid = 0
        self.nid_to_ctx = {}
        self.node_map = {}
        self.metric_id_name_map = {}
        self.number_of_metric_types = 0

        # The meta.db header consists of the common .db header and n sections.
        # We're going to do a little set up work, so that's easy to change if
        # any revisions change the orders.

        # We're are going to specify 2 maps:
        #   - One dictionary maps the section name to an index
        #       (which follows the order that the sections are seen
        #        in the meta.db header)
        #   - The second dictionary maps the section name to a
        #       function that reads the section. Each function is defined
        #       as __read_<section_name>_section(self, section_pointer: int,
        #       section_size: int) -> None

        # Here I'm mapping the section name to it's order in the meta.db header
        header_map = {
            "General Properties": 0,
            "Identifier Names": 1,
            "Performance Metrics": 2,
            "Context Tree": 3,
            "Common String Table": 4,
            "Load Modules": 5,
            "Source Files": 6,
            "Functions": 7,
        }

        # Now let's create a function to section map
        reader_map = {
            "General Properties": self.__read_general_properties_section,
            "Common String Table": self.__read_common_string_table_section,
            "Source Files": self.__read_source_files_section,
            "Functions": self.__read_functions_section,
            "Load Modules": self.__read_load_modules_section,
            "Context Tree": self.__read_context_tree_section,
            "Identifier Names": self.__read_identifier_names_section,
            "Performance Metrics": self.__read_performance_metrics_section,
        }

        # Another thing thing that we should consider is the order to read the sections.
        # Here is a list of section references (x -> y means x references y)
        #   - "Source Files"    ->  "Common String Table"
        #   - "Functions"       ->  "Common String Table"
        #   - "Context Tree"    ->  "Common String Table"
        #   - "Load Modules"    ->  "Common String Table"
        #   - "Functions"       ->  "Source Files"
        #   - "Context Tree"    ->  "Functions"
        #   - "Context Tree"    ->  "Source Files"
        #   - "Functions"       ->  "Load Modules"
        #
        # Thus we want to keep this order when reading sections:
        # "Common String Table" -> "Source Files" -> "Functions" -> "Context Tree", and
        # "Common String Table -> "Load Modules" -> "Context Tree"
        # Here I'm specifying the order of reading the file
        self.read_order = [
            "Common String Table",
            "General Properties",
            "Source Files",
            "Load Modules",
            "Functions",
            "Context Tree",
            "Identifier Names",
            "Performance Metrics",
        ]

        # Let's make sure that we include every section in the read order and reader_map
        assert set(self.read_order) == set(header_map) and set(header_map) == set(
            reader_map
        )

        # Now to the actual reading of the meta.db file

        # reading the meta.db header
        self.__read_common_header()

        # now let's read all the sections
        for section_name in self.read_order:
            section_index = header_map[section_name]
            section_pointer = self.section_pointer[section_index]
            section_size = self.section_size[section_index]
            section_reader = reader_map[section_name]
            section_reader(section_pointer, section_size)

    def get_information_from_context_id(self, context_id: int):
        context: dict = self.context_map[context_id]
        if "string_index" in context:
            return {
                "module": "",
                "file": "",
                "function": self.common_strings[context["string_index"]],
                "relation": -1,
                "lexical_type": -1,
                "line": -1,
                "loop_type": False,
            }

        load_module_index = context["load_module_index"]
        source_file_index = context["source_file_index"]
        source_file_line = context["source_file_line"]
        function_index = context["function_index"]
        lexical_type = context["lexical_type"]
        load_module_offset = context["load_module_offset"]

        source_file_string = None
        module_string = None
        function_string = None
        file_line = -1
        type = None

        if lexical_type == 1:
            # loop construct
            function_string = "loop"
            type = "loop"
        elif lexical_type == 2:
            type = "line"
        elif lexical_type == 3:
            type = "instruction"
        elif lexical_type == 0:
            type = "function"
        else:
            # function is unkown
            function_string = "<unkown function>"

        if load_module_offset is not None:
            load_module_offset = hex(int(load_module_offset))

        if load_module_index is not None:
            load_module = self.load_modules_list[load_module_index]
            module_string = self.common_strings[load_module["string_index"]]
        if source_file_index is not None:
            source_file = self.source_files_list[source_file_index]
            source_file_string = self.common_strings[source_file["string_index"]]
        if function_index is not None:
            function = self.functions_list[function_index]
            function_string = self.common_strings[function["string_index"]]
        if source_file_line is not None:
            file_line = str(source_file_line)
        return {
            "module": module_string,
            "file": source_file_string,
            "function": function_string,
            "line": file_line,
            "type": type,
            "relation": context["relation"],
            "instruction": load_module_offset,
            # "loop_type": loop_type,
        }

    def __read_common_header(self) -> None:
        """
        Reads common .db file header version 4.0
        """

        # read Magic identifier ("HPCPROF-tracedb_")
        # first ten buyes are HPCTOOLKIT in ASCII
        identifier = str(self.file.read(10), encoding=self.encoding)
        assert identifier == "HPCTOOLKIT"

        # next 4 bytes (u8) are the "Specific format identifier"
        format_identifier = str(self.file.read(4), encoding=self.encoding)
        assert format_identifier == "meta"

        # next byte (u8) contains the "Common major version, currently 4"
        self.major_version = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )
        # next byte (u8) contains the "Specific minor version"
        self.minor_version = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )

        self.section_pointer = []
        self.section_size = []
        # In the header each section is given 16 bytes:
        #   - First 8 bytes specify the total size of the section (in bytes)
        #   - Last 8 bytes specify a pointer to the beggining of the section
        for i in range(len(self.read_order)):
            self.section_size.append(
                int.from_bytes(
                    self.file.read(8), byteorder=self.byte_order, signed=self.signed
                )
            )
            self.section_pointer.append(
                int.from_bytes(
                    self.file.read(8), byteorder=self.byte_order, signed=self.signed
                )
            )

    def __read_general_properties_section(
        self, section_pointer: int, section_size: int
    ) -> None:
        """Reads the general properties of the trace.
        Sets:

        self.database_title: Title of the database. May be provided by the user.

        self.database_description: Human-readable Markdown description of the database.
        """

        # go to the right spot in the file
        self.file.seek(section_pointer)
        title_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )
        description_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )

        self.database_title = self.__read_string(title_pointer)
        self.database_description = self.__read_string(description_pointer)

    def __get_common_string(self, string_pointer: int) -> str:
        """Given the file pointer to find string, returns the string."""
        if string_pointer in self.common_string_index_map:
            return self.common_strings[self.common_string_index_map[string_pointer]]
        else:
            return None

    def __read_common_string_table_section(
        self, section_pointer: int, section_size: int
    ) -> None:
        # Let's go to the section
        self.file.seek(section_pointer)

        # We know that this section is just a densely packed list of strings,
        # seperated by the null character
        # So to create a list of these strings, we'll read them all into one string then
        # split them by the null character

        # Reading entire section into a string
        total_section: str = str(self.file.read(section_size), encoding="UTF-8")

        # Splitting entire section into list of strings
        self.common_strings: list[str] = total_section.split("\0")

        # Now we are creating a map between the original location to the string
        # to the index of the string in self.common_strings.
        # This is because we are passed pointers to find the string in other sections
        pointer_index = section_pointer
        # pointer_index = 0
        self.common_string_index_map: dict = {}
        for i in range(len(self.common_strings)):
            self.common_string_index_map[pointer_index] = i
            pointer_index += len(self.common_strings[i]) + 1

    def __get_load_modules_index(self, load_module_pointer: int) -> int:
        """
        Given the pointer to where the file would exists in meta.db,
        returns the index of the file in self.source_files_list.
        """
        return (
            load_module_pointer - self.load_modules_pointer
        ) // self.load_module_size

    def __read_load_modules_section(
        self, section_pointer: int, section_size: int
    ) -> None:
        """
        Reads the "Load Modules" Section of meta.db.
        """
        # go to the right spot in meta.db
        self.file.seek(section_pointer)

        # Load modules used in this database
        self.load_modules_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )
        # Number of load modules listed in this section (u32)
        num_load_modules = int.from_bytes(
            self.file.read(4), byteorder=self.byte_order, signed=self.signed
        )
        # Size of a Load Module Specification, currently 16 (u16)
        self.load_module_size = int.from_bytes(
            self.file.read(2), byteorder=self.byte_order, signed=self.signed
        )

        # Going to store file's path in self.load_modules_list.
        # Each will contain the index of file's path string in
        # self.common_string
        self.load_modules_list: list[dict] = []

        for i in range(num_load_modules):
            current_index = self.load_modules_pointer + (i * self.load_module_size)
            self.file.seek(current_index)

            # Flags -- Reserved for future use (u32)
            # flags = int.from_bytes(
            #     self.file.read(4), byteorder=self.byte_order, signed=self.signed
            # )
            self.file.read(4)
            # empty space that we need to skip
            self.file.read(4)
            # Full path to the associated application binary
            path_pointer = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )
            module_map = {"string_index": self.common_string_index_map[path_pointer]}
            self.load_modules_list.append(module_map)

    def __read_string(self, file_pointer: int) -> str:
        """
        Helper function to read a string from the file starting at the file_pointer
        and ending at the first occurence of the null character
        """
        self.file.seek(file_pointer)
        name = ""
        while True:
            read = str(self.file.read(1), encoding="UTF-8")
            if read == "\0":
                break
            name += read
        return name

    def get_identifier_name(self, kind: int):
        """
        returns the identifier name, given the kind
        """
        return self.identifier_names[kind]

    def __read_identifier_names_section(
        self, section_pointer: int, section_size: int
    ) -> None:
        """
        Reads "Identifier Names" Section and Identifier Name strings in self.names_list
        """

        # go to correct section of file
        self.file.seek(section_pointer)

        # Human-readable names for Identifier kinds
        names_pointer_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )
        # Number of names listed in this section
        num_names = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )

        self.identifier_names: list[str] = []

        for i in range(num_names):
            self.file.seek(names_pointer_pointer + (i * 8))
            names_pointer = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )
            identifier_name = self.__read_string(names_pointer)
            if identifier_name == "NODE":
                self.identifier_names.append("node_pid")
            else:
                self.identifier_names.append(identifier_name.lower())

    def __read_performance_metrics_section(
        self, section_pointer: int, section_size: int
    ) -> None:
        # go to correct section in file
        self.file.seek(section_pointer)

        # {MD} Descriptions of performance metrics
        md_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )

        number_of_md = int.from_bytes(
            self.file.read(4), byteorder=self.byte_order, signed=self.signed
        )

        # Size of the {MD} structure, currently 32
        size_of_md = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )

        # Size of the {PSI} structure, currently 16
        size_of_psi = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )

        # Size of the {SS} structure, currently 24
        self.file.read(1)
        # size_of_ss = int.from_bytes(
        #     self.file.read(1), byteorder=self.byte_order, signed=self.signed
        # )

        # empty space
        self.file.read(1)

        # {PS} Descriptions of propgation scopes.
        # We don't need the descriptions at this point.
        # We will read descriptions for each metric later.
        self.file.read(8)
        # ps_pointer = int.from_bytes(
        #     self.file.read(8), byteorder=self.byte_order, signed=self.signed
        # )

        # Number of propgation scopes
        self.file.read(2)
        # number_of_ps = int.from_bytes(
        #     self.file.read(2), byteorder=self.byte_order, signed=self.signed
        # )

        # Size of the {PS} structure, currently 16
        self.file.read(1)
        # size_of_ps = int.from_bytes(
        #     self.file.read(1), byteorder=self.byte_order, signed=self.signed
        # )

        # Read MD
        for i in range(number_of_md):
            self.file.seek(md_pointer + (size_of_md * i))
            metric_name_pointer = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )

            # {PSI} Instantiated propagated sub-metrics
            psi_pointer = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )

            # {SS} Summary statistics. Hatchet currently don't need summary statistics.
            self.file.read(8)
            # ss_pointer = int.from_bytes(
            #     self.file.read(8), byteorder=self.byte_order, signed=self.signed
            # )

            # Number of instantiated sub-metrics for this metric
            self.number_of_metric_types = int.from_bytes(
                self.file.read(2), byteorder=self.byte_order, signed=self.signed
            )

            # Number of summary statistics for this metric. Hatchet currently
            # don't need summary statistics.
            self.file.read(2)
            # number_of_ss = int.from_bytes(
            #     self.file.read(2), byteorder=self.byte_order, signed=self.signed
            # )

            # empty space
            self.file.read(4)

            metric_name = self.__read_string(metric_name_pointer)

            # Instantiated propagated sub-metrics (PSI)
            # Copied from FORMATS.md: Propagated metric values are generated for each
            # context by summing values attributed to children contexts, within the
            # measurements for a single application thread. Which children are included
            # in this sum is indicated by the *pScope PS structure.
            for i in range(self.number_of_metric_types):
                self.file.seek(psi_pointer + (size_of_psi * i))

                ps_in_psi_pointer = int.from_bytes(
                    self.file.read(8), byteorder=self.byte_order, signed=self.signed
                )

                psi_metric_id = int.from_bytes(
                    self.file.read(2), byteorder=self.byte_order, signed=self.signed
                )

                # empty space
                self.file.read(6)

                # Get the description of propagation scopes for this metric ID.
                self.file.seek(ps_in_psi_pointer)

                # Examples: point, function, lex_aware, execution
                psi_name_pointer = int.from_bytes(
                    self.file.read(8), byteorder=self.byte_order, signed=self.signed
                )

                # Differet types:
                # 1: point, 2: execution, 3: function, 4: lex-aware.
                self.file.read(1)
                # psi_type = int.from_bytes(
                #     self.file.read(1), byteorder=self.byte_order, signed=self.signed
                # )

                self.file.read(1)
                # psi_index = int.from_bytes(
                #     self.file.read(1), byteorder=self.byte_order, signed=self.signed
                # )

                # empty space
                self.file.read(6)

                psi_name = self.__read_string(psi_name_pointer)
                self.metric_id_name_map[psi_metric_id] = (metric_name, psi_name)

    def __get_function_index(self, function_pointer: int) -> int:
        """
        Given the pointer to where the function would exists in meta.db,
        returns the index of the file in self.functions_list.
        """
        index = (function_pointer - self.functions_array_pointer) // self.function_size
        assert index < len(self.functions_list)
        return index

    def __read_functions_section(self, section_pointer: int, section_size: int) -> None:
        """
        Reads the "Functions" section of meta.db.
        """

        # go to correct section in file
        self.file.seek(section_pointer)

        self.functions_array_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )
        num_functions = int.from_bytes(
            self.file.read(4), byteorder=self.byte_order, signed=self.signed
        )
        self.function_size = int.from_bytes(
            self.file.read(2), byteorder=self.byte_order, signed=self.signed
        )

        self.functions_list: list[dict] = []
        for i in range(num_functions):
            current_index = self.functions_array_pointer + (i * self.function_size)
            self.file.seek(current_index)
            function_name_pointer = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )
            modules_pointer = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )
            modules_offset = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )
            file_pointer = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )
            source_line = int.from_bytes(
                self.file.read(4), byteorder=self.byte_order, signed=self.signed
            )
            # flags = int.from_bytes(
            #     self.file.read(4), byteorder=self.byte_order, signed=self.signed
            # )
            self.file.read(4)
            source_file_index = None
            load_module_index = None
            function_name_index = None
            if function_name_pointer != 0:
                function_name_index = self.common_string_index_map[
                    function_name_pointer
                ]
            if modules_pointer != 0:
                load_module_index = self.__get_load_modules_index(modules_pointer)
                # currently ignoring offset -- no idea how that's used
            if file_pointer != 0:
                source_file_index = self.__get_source_file_index(file_pointer)

            current_function_map = {
                "string_index": function_name_index,
                "source_line": source_line,
                "load_modules_index": load_module_index,
                "source_file_index": source_file_index,
                "load_modules_offset": modules_offset,
            }
            self.functions_list.append(current_function_map)

    def __get_source_file_index(self, source_file_pointer: int) -> int:
        """
        Given the pointer to where the file would exists in meta.db,
        returns the index of the file in self.source_files_list.
        """
        index = (
            source_file_pointer - self.source_files_pointer
        ) // self.source_file_size
        assert index < len(self.source_files_list)
        return index

    def __read_source_files_section(
        self, section_pointer: int, section_size: int
    ) -> None:
        """
        Reads the "Source Files" Section of meta.db.
        """

        self.file.seek(section_pointer)

        # Source files used in this database
        self.source_files_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )

        # Number of source files listed in this section (u32)
        num_files = int.from_bytes(
            self.file.read(4), byteorder=self.byte_order, signed=self.signed
        )

        # Size of a Source File Specification, currently 16 (u16)
        self.source_file_size = int.from_bytes(
            self.file.read(2), byteorder=self.byte_order, signed=self.signed
        )

        # Looping through individual files to get there information now
        self.file.seek(self.source_files_pointer)

        # Going to store file's path in self.files_list.
        # Each will contain the index of file's path string in
        # self.common_string
        self.source_files_list: list[dict] = []
        for i in range(num_files):
            # Reading information about each individual source file
            self.file.seek(self.source_files_pointer + (i * self.source_file_size))

            # flag = int.from_bytes(
            #     self.file.read(4), byteorder=self.byte_order, signed=self.signed
            # )
            self.file.read(4)
            # empty space that we need to skip
            self.file.read(4)
            # Path to the source file. Absolute, or relative to the root database
            # directory. The string pointed to by pPath is completely within the
            # Common String Table section, including the terminating NUL byte.
            file_path_pointer = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )
            string_index = self.common_string_index_map[file_path_pointer]
            source_file_map = {"string_index": string_index}
            self.source_files_list.append(source_file_map)

    def __read_context_tree_section(
        self, section_pointer: int, section_size: int
    ) -> None:
        """
        Reads the "Context Tree" section of meta.db.

        Loops and calls __read_single_entry_point with the correct pointer to read
        the correct entry and add it to the CCT.
        """

        self.roots = []
        self.context_map: dict[int, dict] = {}

        # Reading "Context Tree" section header

        # make sure we're in the right spot of the file
        self.file.seek(section_pointer)

        # ({Entry}[nEntryPoints]*)
        entry_points_array_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )
        # (u16)
        num_entry_points = int.from_bytes(
            self.file.read(2), byteorder=self.byte_order, signed=self.signed
        )
        # (u8)
        entry_point_size = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )

        for i in range(num_entry_points):
            current_pointer = entry_points_array_pointer + (i * entry_point_size)
            self.__read_single_entry_point(current_pointer)

    def __read_single_entry_point(self, entry_point_pointer: int) -> None:
        """
        Reads single (root) context entry.

        Reads the correct entry and adds it to the CCT.
        """

        self.file.seek(entry_point_pointer)

        # Reading information about child contexts
        # Total size of *pChildren (I call pChildren children_pointer), in bytes (u64)
        children_size = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )
        # Pointer to the array of child contexts
        children_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )

        # Reading information about this context
        # Unique identifier for this context (u32)
        context_id = int.from_bytes(
            self.file.read(4), byteorder=self.byte_order, signed=self.signed
        )
        # Type of entry point used here (u16)
        # entry_point_type = int.from_bytes(
        #     self.file.read(2), byteorder=self.byte_order, signed=self.signed
        # )
        self.file.read(2)
        # next 2 bytes are blank
        self.file.read(2)
        # Human-readable name for the entry point
        pretty_name_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )

        # map context for this context
        string_index = self.common_string_index_map[pretty_name_pointer]

        context = {
            "relation": -1,
            "lexical_type": -1,
            "function_index": -1,
            "source_file_index": -1,
            "source_file_line": -1,
            "load_module_index": -1,
            "load_module_offset": -1,
            "string_index": string_index,
        }
        self.context_map[context_id] = context
        # Create Node for this context
        root_context_info = self.get_information_from_context_id(context_id)
        node = Node(
            Frame({"type": "function", "name": root_context_info["function"]}),
            parent=None,
            hnid=self._add_context_id(context_id),
        )

        # Adding the Node to the CCT
        self.roots.append(node)
        self.node_map[context_id] = node

        # Reading the children contexts
        self.__read_children_contexts(children_pointer, children_size, node, context_id)

    def __read_children_contexts(
        self,
        context_array_pointer: int,
        total_size: int,
        parent_node: Node,
        parent_context_id: int,
    ) -> None:
        """
        Recursive function to read all child contexts and add it to the CCT
        """

        if total_size <= 0 or context_array_pointer <= 0:
            return
        self.file.seek(context_array_pointer)
        index = 0
        while index < total_size:
            # Reading information about child contexts (as in the children of
            # this context)
            # Total size of *pChildren (I call pChildren children_pointer),
            # in bytes (u64)
            children_size = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )
            index += 8
            # Pointer to the array of child contexts
            children_pointer = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )
            index += 8

            # Reading information about this context
            # Unique identifier for this context (u32)
            context_id = int.from_bytes(
                self.file.read(4), byteorder=self.byte_order, signed=self.signed
            )
            index += 4
            # Reading flags (u8)
            flags = int.from_bytes(
                self.file.read(1), byteorder=self.byte_order, signed=self.signed
            )
            index += 1
            # Relation this context has with its parent (u8)
            relation = int.from_bytes(
                self.file.read(1), byteorder=self.byte_order, signed=self.signed
            )
            index += 1
            # Type of lexical context represented (u8)
            lexical_type = int.from_bytes(
                self.file.read(1), byteorder=self.byte_order, signed=self.signed
            )
            index += 1
            # Size of flex, in u8[8] "words" (bytes / 8) (u8)
            num_flex_words = int.from_bytes(
                self.file.read(1), byteorder=self.byte_order, signed=self.signed
            )
            index += 1
            # Bitmask for defining propagation scopes (u16)
            # propogation = int.from_bytes(
            #     self.file.read(2), byteorder=self.byte_order, signed=self.signed
            # )
            self.file.read(2)
            index += 2
            # Empty space
            self.file.read(6)
            index += 6

            # reading flex
            flex = self.file.read(8 * num_flex_words)
            index += 8 * num_flex_words

            function_index: int = None
            source_file_index: int = None
            source_file_line: int = None
            load_module_index: int = None
            load_module_offset: int = None

            # flex is u8[8][num_flex],
            # meaning that one flex word is 8 bytes or u64

            # Bit 0: hasFunction. If 1, the following sub-fields of flex are present:
            #   flex[0]: FS* pFunction: Function associated with this context
            if flags & 1 != 0:
                sub_flex = int.from_bytes(
                    flex[0:8], byteorder=self.byte_order, signed=self.signed
                )
                flex = flex[8:]
                function_index = self.__get_function_index(sub_flex)

            # Bit 1: hasSrcLoc. If 1, the following sub-fields of flex are present:
            #   flex[1]: SFS* pFile: Source file associated with this context
            #   flex[2]: u32 line: Associated source line in pFile
            if flags & 2 != 0:
                sub_flex_1 = int.from_bytes(
                    flex[0:8], byteorder=self.byte_order, signed=self.signed
                )
                sub_flex_2 = int.from_bytes(
                    flex[8:10], byteorder=self.byte_order, signed=self.signed
                )
                flex = flex[10:]
                source_file_index = self.__get_source_file_index(sub_flex_1)
                source_file_line = sub_flex_2

            # Bit 2: hasPoint. If 1, the following sub-fields of flex are present:
            #   flex[3]: LMS* pModule: Load module associated with this context
            #   flex[4]: u64 offset: Associated byte offset in *pModule
            if flags & 4 != 0:
                sub_flex_1 = int.from_bytes(
                    flex[0:8], byteorder=self.byte_order, signed=self.signed
                )
                sub_flex_2 = int.from_bytes(
                    flex[8:16], byteorder=self.byte_order, signed=self.signed
                )
                flex = flex[16:]
                load_module_index = self.__get_load_modules_index(sub_flex_1)
                load_module_offset = sub_flex_2

            if lexical_type == 0:
                # function call
                # this means that information about the
                # source file and module are with the parent
                parent_information = self.context_map[parent_context_id]
                if "string_index" in parent_information and function_index is not None:
                    # This means that the parent is the root, and it's
                    # information is useless
                    function = self.functions_list[function_index]
                    # Getting source file line
                    source_file_line = function["source_line"]
                    # Getting source file name
                    source_file_index = function["source_file_index"]
                    # Getting Load Module Name
                    load_module_index = function["load_modules_index"]
                    # Getting Load Module Offset

                    load_module_offset = function["load_modules_offset"]
                else:
                    if source_file_index is None:
                        source_file_index = parent_information["source_file_index"]
                    if source_file_line is None:
                        source_file_line = parent_information["source_file_line"]
                    if load_module_index is None:
                        load_module_index = parent_information["load_module_index"]
                        load_module_offset = parent_information["load_module_offset"]

            # creating a map for this context
            context = {
                "relation": relation,
                "lexical_type": lexical_type,
                "function_index": function_index,
                "source_file_index": source_file_index,
                "source_file_line": source_file_line,
                "load_module_index": load_module_index,
                "load_module_offset": load_module_offset,
            }

            self.context_map[context_id] = context
            node_context_info = self.get_information_from_context_id(context_id)
            node = None
            if node_context_info["type"] == "line":
                node = Node(
                    Frame(
                        {
                            "type": "line",
                            "file": node_context_info["file"],
                            "line": node_context_info["line"],
                        }
                    ),
                    parent=parent_node,
                    hnid=self._add_context_id(context_id),
                )
            elif node_context_info["type"] == "instruction":
                node = Node(
                    Frame(
                        {
                            "type": "instruction",
                            "module": node_context_info["module"],
                            "instruction": node_context_info["instruction"],
                        }
                    ),
                    parent=parent_node,
                    hnid=self._add_context_id(context_id),
                )
            elif node_context_info["type"] == "loop":
                node = Node(
                    Frame(
                        {
                            "type": "loop",
                            "file": node_context_info["file"],
                            "line": node_context_info["line"],
                        }
                    ),
                    parent=parent_node,
                    hnid=self._add_context_id(context_id),
                )
            else:
                node = Node(
                    Frame({"type": "function", "name": node_context_info["function"]}),
                    parent=parent_node,
                    hnid=self._add_context_id(context_id),
                )

            # Connecting this node to the parent node
            parent_node.add_child(node)
            next_parent_node = node

            # Adding this node to the graph
            self.node_map[context_id] = node

            # recursively call this function to add more children
            return_address = self.file.tell()
            self.__read_children_contexts(
                children_pointer, children_size, next_parent_node, context_id
            )
            self.file.seek(return_address)


class ProfileReader:
    # class to read self.data from profile.db file

    def __init__(self, file_location, meta_reader):
        # gets the pi_ptr variable to be able to read the identifier tuples
        self.meta_reader: MetaReader = meta_reader

        # open the file to ready in binary mode (rb)
        self.file = open(file_location, "rb")

        # setting necessary read options
        self.byte_order = "little"
        self.signed = False
        self.encoding = "ASCII"

        # The profile.db header consists of the common .db header and n sections.
        # We're going to do a little set up work, so that's easy to change if
        # any revisions change the orders.

        # We're are going to specify 2 maps:
        #   - One dictionary maps the section name to an index
        #       (which follows the order that the sections are seen
        #        in the meta.db header)
        #   - The second dictionary maps the section name to a
        #       function that reads the section. Each function is defined
        #       as __read_<section_name>_section(self, section_pointer: int,
        #       section_size: int) -> None

        # Here I'm mapping the section name to it's order in the meta.db header
        header_map = {"Profiles Information": 0, "Hierarchical Identifier Tuples": 1}

        # Now let's create a function to section map
        reader_map = {
            "Profiles Information": self.__read_profiles_information_section,
            "Hierarchical Identifier Tuples": self.__read_hit_section,
        }

        # Another thing thing that we should consider is the order to read the sections.
        # Here is a list of section references (x -> y means x references y)
        #   - "Profiles Information" -> "Hierarchical Identifier Tuples"
        #
        # Thus we want to keep this order when reading sections:
        # "Profiles Information" -> "Hierarchical Identifier Tuples"
        # Here I'm specifying the order of reading the file
        self.read_order = ["Hierarchical Identifier Tuples", "Profiles Information"]

        # Let's make sure that we include every section in the read order and reader_map
        assert set(self.read_order) == set(header_map) and set(header_map) == set(
            reader_map
        )

        # Now to the actual reading of the meta.db file

        # reading the meta.db header
        self.__read_common_header()

        # now let's read all the sections
        for section_name in self.read_order:
            section_index = header_map[section_name]
            section_pointer = self.section_pointer[section_index]
            section_size = self.section_size[section_index]
            section_reader = reader_map[section_name]
            section_reader(section_pointer, section_size)

    def __read_profiles_information_section(
        self, section_pointer: int, section_size: int
    ) -> None:
        """
        Reads Profile Information section.
        """

        self.file.seek(section_pointer)

        # Description for each profile (u64)
        profiles_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )
        # Number of profiles listed in this section (u32)
        num_profiles = int.from_bytes(
            self.file.read(4), byteorder=self.byte_order, signed=self.signed
        )
        # Size of a {PI} structure, currently 40 (u8)
        profile_size = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )

        self.profile_info_list = []
        for i in range(num_profiles):
            file_index = profiles_pointer + (i * profile_size)
            self.file.seek(file_index)

            # Profile-Major Sparse Value Block.
            # We don't need to read this section unless we want
            # the summary statistics. The commented out lines below
            # show how to read PSVB in case we read it in future.
            self.file.read(0x20)
            # psvb = self.file.read(0x20)

            # Identifier tuple for this profile
            hit_pointer = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )
            # (u32)
            flags = int.from_bytes(
                self.file.read(4), byteorder=self.byte_order, signed=self.signed
            )
            profile_map = {"hit_pointer": hit_pointer, "flags": flags}

            self.profile_info_list.append(profile_map)
            if hit_pointer == 0:
                # this is a summary profile
                self.summary_profile_index = i

    def get_hit_from_profile(self, index: int) -> list:
        profile = self.profile_info_list[index]
        hit_pointer = profile["hit_pointer"]
        if hit_pointer != 0:
            return self.hit_map[hit_pointer]
        else:
            profile = self.profile_info_list[self.summary_profile_index]
            hit_pointer = profile["hit_pointer"]
            return self.hit_map[hit_pointer]

    def __read_hit_section(self, section_pointer: int, section_size: int) -> None:
        """
        Reads Hierarchical Identifier Tuples section of profile.db
        """
        # let's get to the correct spot in the file
        self.file.seek(section_pointer)

        self.hit_map = {}

        while (self.file.tell() - section_pointer) < section_size:
            # hit pointer
            hit_pointer = self.file.tell()

            # Number of identifications in this tuple (u16)
            num_tuples = int.from_bytes(
                self.file.read(2), byteorder=self.byte_order, signed=self.signed
            )

            # empty space
            self.file.read(6)

            # Identifications for an application thread
            # Read H.I.T.s
            profile_properties = {}
            for i in range(num_tuples):
                # One of the values listed in the meta.db Identifier Names section. (u8)
                kind = int.from_bytes(
                    self.file.read(1), byteorder=self.byte_order, signed=self.signed
                )
                # empty space
                self.file.read(1)

                # flag
                flags = int.from_bytes(
                    self.file.read(2), byteorder=self.byte_order, signed=self.signed
                )
                # self.file.read(2)

                # Logical identifier value, may be arbitrary but dense towards 0. (u32)
                logical_id = int.from_bytes(
                    self.file.read(4), byteorder=self.byte_order, signed=self.signed
                )
                # self.file.read(4)

                # Physical identifier value, eg. hostid or PCI bus index. (u64)
                physical_id = int.from_bytes(
                    self.file.read(8), byteorder=self.byte_order, signed=self.signed
                )

                identifier_name = self.meta_reader.get_identifier_name(kind)

                if flags & 0x1:
                    profile_properties[identifier_name] = physical_id
                else:
                    profile_properties[identifier_name] = logical_id
                # tuples_list.append((identifier_name, physical_id, flags, logical_id))
            self.hit_map[hit_pointer] = profile_properties

    def __read_common_header(self) -> None:
        """
        Reads common .db file header version 4.0
        """

        # read Magic identifier ("HPCPROF-tracedb_")
        # first ten buyes are HPCTOOLKIT in ASCII
        identifier = str(self.file.read(10), encoding=self.encoding)
        assert identifier == "HPCTOOLKIT"

        # next 4 bytes (u8) are the "Specific format identifier"
        format_identifier = str(self.file.read(4), encoding=self.encoding)
        assert format_identifier == "prof"

        # next byte (u8) contains the "Common major version, currently 4"
        self.major_version = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )
        # next byte (u8) contains the "Specific minor version"
        self.minor_version = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )

        self.section_pointer = []
        self.section_size = []
        # In the header each section is given 16 bytes:
        #   - First 8 bytes specify the total size of the section (in bytes)
        #   - Last 8 bytes specify a pointer to the beggining of the section
        for i in range(len(self.read_order)):
            self.section_size.append(
                int.from_bytes(
                    self.file.read(8), byteorder=self.byte_order, signed=self.signed
                )
            )
            self.section_pointer.append(
                int.from_bytes(
                    self.file.read(8), byteorder=self.byte_order, signed=self.signed
                )
            )


class DefaultReader:
    """Reads the defaults.yaml file in the metrics directory.
    This file is needed to obtain inclusive and exclusive metrics."""

    def __init__(self, file_location: str, meta_reader: MetaReader) -> None:
        self.meta_reader = meta_reader
        self.metric_defaults = {}

        with open(file_location, "rb") as self.file:
            file_contents = yaml.safe_load(self.file)

            # Gets the inclusive and exclusive types for each metric.
            # This part might need modifications in future.
            # Reads the metric information in default.yaml and gets the
            # inclusive and exclusive metrics by looking at corresponding
            # section (inclusive/standard and exlusive/standard) of each
            # metric. Inclusive metric is usually "execution",
            # exclusive metric is usually "function".
            for metric in file_contents["roots"]:
                self.metric_defaults[metric["name"]] = {}
                if "Sum" in metric["variants"]:
                    if "percent" in metric["variants"]["Sum"]["render"]:
                        formula = metric["variants"]["Sum"]["formula"]
                        if "inclusive" in formula:
                            self.metric_defaults[metric["name"]]["inclusive"] = formula[
                                "inclusive"
                            ]["standard"]["scope"]

                        if "exclusive" in formula:
                            self.metric_defaults[metric["name"]]["exclusive"] = formula[
                                "exclusive"
                            ]["standard"]["scope"]

                # TODO: Not sure how to get inclusive and exclusive metrics when
                # we have a list in the standard section instead of a single variable.
                elif "" in metric["variants"]:
                    if "percent" in metric["variants"][""]["render"]:
                        formula = metric["variants"][""]["formula"]
                        if "inclusive" in formula:
                            self.metric_defaults[metric["name"]]["inclusive"] = formula[
                                "inclusive"
                            ]["standard"]["scope"]

                        if "exclusive" in formula:
                            self.metric_defaults[metric["name"]]["exclusive"] = formula[
                                "exclusive"
                            ]["standard"]["scope"]


class CCTReader:
    """Reads the metric and profile information (thread, rank, etc.)
    for each context."""

    def __init__(
        self,
        file_location: str,
        meta_reader: MetaReader,
        profile_reader: ProfileReader,
        defaults_reader: DefaultReader,
    ) -> None:
        # open file
        self.file = open(file_location, "rb")
        self.meta_reader = meta_reader
        self.profile_reader = profile_reader
        self.defaults_reader = defaults_reader
        self.node_dicts = {}
        self.inclusive_metrics = set([])
        self.exclusive_metrics = set([])

        # setting necessary read options
        self.byte_order = "little"
        self.signed = False
        self.encoding = "ASCII"

        # The cct.db header consists of the common .db header and n sections.
        # We're going to do a little set up work, so that's easy to change if
        # any revisions change the orders.

        # We're are going to specify 2 maps:
        #   - One dictionary maps the section name to an index
        #       (which follows the order that the sections are seen
        #        in the meta.db header)
        #   - The second dictionary maps the section name to a
        #       function that reads the section. Each function is defined
        #       as __read_<section_name>_section(self, section_pointer: int,
        #       section_size: int) -> None

        # Here I'm mapping the section name to it's order in the meta.db header
        header_map = {"Context Headers": 0}

        # Now let's create a function to section map
        reader_map = {"Context Headers": self.__read_cct_info_section}

        # Another thing thing that we should consider is the order to read the sections.
        # Here I'm specifying the order of reading the file
        self.read_order = ["Context Headers"]

        # Let's make sure that we include every section in the read order and reader_map
        assert set(self.read_order) == set(header_map) and set(header_map) == set(
            reader_map
        )

        # Now to the actual reading of the meta.db file

        # reading the meta.db header
        self.__read_common_header()

        # now let's read all the sections
        for section_name in self.read_order:
            section_index = header_map[section_name]
            section_pointer = self.section_pointer[section_index]
            section_size = self.section_size[section_index]
            section_reader = reader_map[section_name]
            section_reader(section_pointer, section_size)

    def __read_common_header(self) -> None:
        """
        Reads common .db file header version 4.0
        """

        # read Magic identifier ("HPCPROF-tracedb_")
        # first ten buyes are HPCTOOLKIT in ASCII
        identifier = str(self.file.read(10), encoding=self.encoding)
        assert identifier == "HPCTOOLKIT"

        # next 4 bytes (u8) are the "Specific format identifier"
        format_identifier = str(self.file.read(4), encoding=self.encoding)
        assert format_identifier == "ctxt"

        # next byte (u8) contains the "Common major version, currently 4"
        self.major_version = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )
        # next byte (u8) contains the "Specific minor version"
        self.minor_version = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )

        self.section_pointer = []
        self.section_size = []
        # In the header each section is given 16 bytes:
        #   - First 8 bytes specify the total size of the section (in bytes)
        #   - Last 8 bytes specify a pointer to the beggining of the section
        for i in range(len(self.read_order)):
            self.section_size.append(
                int.from_bytes(
                    self.file.read(8), byteorder=self.byte_order, signed=self.signed
                )
            )
            self.section_pointer.append(
                int.from_bytes(
                    self.file.read(8), byteorder=self.byte_order, signed=self.signed
                )
            )

    def __create_node_dict(
        self,
        identifier,
        profile_info,
        node_name,
        node,
        context_info,
        metric_name=False,
        value=False,
    ) -> None:
        """Stores profile and metric information for each context.
        node_dicts will be used later to create the dataframe."""
        self.node_dicts[identifier] = dict(profile_info)
        self.node_dicts[identifier]["name"] = node_name
        self.node_dicts[identifier]["node"] = node
        self.node_dicts[identifier].update(context_info)
        if metric_name:
            self.node_dicts[identifier][metric_name] = value

    def __read_cct_info_section(self, section_pointer: int, section_size: int) -> None:
        """
        Reader Context Trace Headers section of trace.db
        """
        # get to the right place in the file
        self.file.seek(section_pointer)

        # Header for each trace (u64)
        context_pointer = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )
        # Number of traces listed in this section (u32)
        self.file.read(4)
        # num_contexts = int.from_bytes(
        #     self.file.read(4), byteorder=self.byte_order, signed=self.signed
        # )
        # Size of a {CI} structure, currently 32
        context_size = int.from_bytes(
            self.file.read(1), byteorder=self.byte_order, signed=self.signed
        )

        # Read metric and profile information for each context.
        for i in self.meta_reader.context_map.keys():
            # get the corresponding Hatchet node for the HPCToolkit context.
            node = self.meta_reader.node_map[i]

            header_pointer = context_pointer + ((i) * context_size)
            self.__read_single_context_value_block(
                header_pointer,
                node,
                self.meta_reader.get_information_from_context_id(i),
            )

    def __read_single_context_value_block(
        self, header_pointer: int, node: Node, context_info: dict
    ) -> None:
        """
        Reads the value block that includes all the metrics and
        profile information (rank, thread, etc.) of a context.
        """

        def __read_profiles(
            next_metric_index,
            visited_profiles,
            node_name,
            node,
            context_info,
            metric_name,
        ):
            """
            Iterates over the profiles and corresponding metric values
            of a metric associated with a context.
            For example, in the first iteration, we focus on 'context1,'
            and within that context, we examine 'metric1.' Within
            'metric1,' we start with 'rank 0, thread 0,' and retrieve
            its associated value. In the second iteration of the while
            loop, we move to 'rank 0, thread 1' and retrieve its value.
            """
            while self.file.tell() < next_metric_index:
                prof_index = int.from_bytes(
                    self.file.read(4), byteorder=self.byte_order, signed=self.signed
                )
                value = struct.unpack("d", self.file.read(8))[0]

                # get the profile info by using its index.
                profile_info = self.profile_reader.get_hit_from_profile(prof_index)

                # keep track of the profiles that has no values. we need to
                # manually create these profiles to convert HPCToolkit's
                # sparse representation to the Hatchet representation.
                visited_profiles.add(prof_index)

                # Example: (node, rank 0, thread 0)
                identifier = (node,) + tuple(profile_info.values())

                # add the metric value if the identifier has seen
                # before. Otherwise, add new information.
                if identifier in self.node_dicts.keys():
                    self.node_dicts[identifier][metric_name] = value
                else:
                    self.__create_node_dict(
                        identifier,
                        profile_info,
                        node_name,
                        node,
                        context_info,
                        metric_name,
                        value,
                    )

        self.file.seek(header_pointer)

        # Header for each trace (u64)
        nonzero_vals = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )

        # Stores the profile and corresponding metric value.
        profile_value_pairs = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )

        # Number of non-empty (i.e. non-zero) metrics.
        nonempty_metrics = int.from_bytes(
            self.file.read(2), byteorder=self.byte_order, signed=self.signed
        )

        # empty space
        self.file.read(6)

        metrics_to_index = int.from_bytes(
            self.file.read(8), byteorder=self.byte_order, signed=self.signed
        )

        metric_indices = []
        self.file.seek(metrics_to_index)
        for _ in range(nonempty_metrics):
            metric_id = int.from_bytes(
                self.file.read(2), byteorder=self.byte_order, signed=self.signed
            )
            start_index = int.from_bytes(
                self.file.read(8), byteorder=self.byte_order, signed=self.signed
            )
            metric_indices.append((metric_id, start_index))

        # construct the node name for the context. it will be
        # used to create the name column in the dataframe.
        node_name = None
        if "type" in context_info.keys():
            if context_info["type"] == "function":
                node_name = context_info["function"]
            # file:line
            elif context_info["type"] == "line":
                node_name = "{}:{}".format(context_info["file"], context_info["line"])
            # Loop@file:line
            elif context_info["type"] == "loop":
                node_name = "Loop@{}:{}".format(
                    context_info["file"], context_info["line"]
                )
            # module:instruction
            # Example: app.exe:0x1581
            elif context_info["type"] == "instruction":
                node_name = "{}:{}".format(
                    context_info["module"], context_info["instruction"]
                )
        # the root node, which is the parent of the main function,
        # don't have a type. its name is stored in the 'function' key.
        # that's why we create a 'type' for it.
        else:
            node_name = context_info["function"]
            context_info["type"] = "function"
            context_info["module"] = None
            context_info["file"] = None
            context_info.pop("loop_type", None)
            context_info.pop("lexical_type", None)

        # I encountered a context in which the 'function' was set to None:
        # {'module': '/usr/lib64/libmlx5.so.1.24.44.0',
        # 'file': '[libucs.so.0.0.0]',
        # 'function': None,
        # 'line': '-1',
        # 'type': 'function',
        # 'relation': 1,
        # 'instruction': '-0x1'}
        # I believe the type should have been 'instruction' instead of 'function'.
        # That might be a bug on the HPCToolkit side.
        if node_name is None:
            node_name = "{}:{}".format(
                context_info["module"], context_info["instruction"]
            )

        # add name to the node's frame.
        node.frame.attrs["name"] = node_name

        # we need to keep track of the visited profiles. HPCToolkit
        # sparse representation doesn't provide profile information
        # if it has only 0 values. We need to create them manually
        # to be able to show those profiles (i.e. rank/thread) on
        # the dataframe.
        visited_profiles = set([])
        not_visited_profiles = list(self.profile_reader.profile_info_list)

        # iterate over each metric.
        # context1 -> metric1, context1 -> metric2, ...
        for i in range(len(metric_indices)):
            metric_id = metric_indices[i][0]
            start_index = metric_indices[i][1]
            next_metric_index = None

            # find the correct location to read.
            # a profile and its value takes 12 bytes.
            if i != len(metric_indices) - 1:
                next_metric_index = profile_value_pairs + (
                    (metric_indices[i + 1][1]) * 12
                )
            else:
                next_metric_index = profile_value_pairs + ((nonzero_vals) * 12)
            self.file.seek(profile_value_pairs + (start_index * 12))

            # get the metric name and type (point, execution, function, lex-aware)
            # from the corresponding id.
            metric = self.meta_reader.metric_id_name_map[metric_id][0]
            metric_type = self.meta_reader.metric_id_name_map[metric_id][1]

            # get the default inclusive and exclusive metric type
            # for the corresponding metric.
            exc, inc = None, None
            if metric in self.defaults_reader.metric_defaults.keys():
                exc = self.defaults_reader.metric_defaults[metric]["exclusive"]
                inc = self.defaults_reader.metric_defaults[metric]["inclusive"]

            # Read profiles and values for the current metric.
            if metric_type == inc:
                metric = metric + " (inc)"
                self.inclusive_metrics.add(metric)
            elif metric_type == exc:
                self.exclusive_metrics.add(metric)
            __read_profiles(
                next_metric_index,
                visited_profiles,
                node_name,
                node,
                context_info,
                metric,
            )

        # remove the visited profiles.
        not_visited_profiles = [
            i for j, i in enumerate(not_visited_profiles) if j not in visited_profiles
        ]

        # iterate over the not visited nodes and create dummy instances.
        for profile in not_visited_profiles:
            hit_pointer = profile["hit_pointer"]
            if hit_pointer != 0:
                dummy_profile_info = self.profile_reader.hit_map[hit_pointer]
                dummy_identifier = (node,) + tuple(dummy_profile_info.values())

                self.__create_node_dict(
                    dummy_identifier,
                    dummy_profile_info,
                    node_name,
                    node,
                    context_info,
                    metric_name=False,
                    value=False,
                )


class HPCToolkitV4Reader:
    def __init__(self, directory: str) -> None:
        self.directory = directory

    def read(self):
        self.meta_reader: MetaReader = MetaReader(self.directory + "/meta.db")
        self.profile_reader: ProfileReader = ProfileReader(
            self.directory + "/profile.db", self.meta_reader
        )
        self.defaults_reader: DefaultReader = DefaultReader(
            self.directory + "/metrics/default.yaml", self.meta_reader
        )
        self.cct_reader = CCTReader(
            self.directory + "/cct.db",
            self.meta_reader,
            self.profile_reader,
            self.defaults_reader,
        )

        return self.create_graphframe()

    def create_graphframe(self) -> GraphFrame:
        graph = Graph(self.meta_reader.roots)
        graph.enumerate_traverse()
        dataframe = pd.DataFrame.from_dict(data=self.cct_reader.node_dicts.values())

        columns = dataframe.columns
        if "thread" in columns and "rank" in columns:
            indices = ["node", "rank", "thread"]
        elif "rank" in columns:
            indices = ["node", "rank"]
        elif "thread" in columns:
            indices = ["node", "thread"]
        else:
            indices = ["node"]

        dataframe.set_index(indices, inplace=True)
        # create missing ranks/threads. this adds NaN to each column for
        # the added rank/thread.
        new_index = pd.MultiIndex.from_product(dataframe.index.levels)
        dataframe = dataframe.reindex(new_index)
        dataframe.sort_index(inplace=True)

        self.cct_reader.exclusive_metrics = list(self.cct_reader.exclusive_metrics)
        self.cct_reader.inclusive_metrics = list(self.cct_reader.inclusive_metrics)

        default_metric = "time"
        if "CPUTIME (sec)" in dataframe.columns:
            dataframe.rename(
                columns={
                    "CPUTIME (sec)": "time",
                    "CPUTIME (sec) (inc)": "time (inc)",
                },
                inplace=True,
            )
            self.cct_reader.exclusive_metrics.remove("CPUTIME (sec)")
            self.cct_reader.inclusive_metrics.remove("CPUTIME (sec) (inc)")
            self.cct_reader.exclusive_metrics.append("time")
            self.cct_reader.inclusive_metrics.append("time (inc)")
        else:
            default_metric = self.cct_reader.exclusive_metrics[0]

        columns_to_fill = [
            col
            for col in dataframe.columns
            if col not in self.cct_reader.exclusive_metrics
            and col not in self.cct_reader.inclusive_metrics
        ]

        # fills the missing rows. Example:
        # main_node, rank0, thread0 -> NaN
        # main_node, rank0, thread1 -> main
        # main_node, rank0, thread2 -> NaN
        # Groups by (main_node, rank0) and checks
        # the other rows in the group.
        # 'ffill' fills the NaN value of thread2.
        # 'bfill' fills the NaN value if thread0.
        for i in range(1, len(indices)):
            dataframe[columns_to_fill] = dataframe.groupby(indices[:-i])[
                columns_to_fill
            ].transform(lambda x: x.ffill().bfill())
            if not dataframe.isna().values.any():
                break

        # if the metric is numeric, fill NaNs with zero.
        dataframe[self.cct_reader.inclusive_metrics] = dataframe[
            self.cct_reader.inclusive_metrics
        ].fillna(0)
        dataframe[self.cct_reader.exclusive_metrics] = dataframe[
            self.cct_reader.exclusive_metrics
        ].fillna(0)

        if "line" in dataframe.columns:
            dataframe["line"] = dataframe["line"].astype("int64")
        if "core" in dataframe.columns:
            dataframe["core"] = dataframe["core"].astype("int64")
        if "node_pid" in dataframe.columns:
            dataframe["node_pid"] = dataframe["node_pid"].astype("int64")

        return hatchet.graphframe.GraphFrame(
            graph,
            dataframe,
            exc_metrics=self.cct_reader.exclusive_metrics,
            inc_metrics=self.cct_reader.inclusive_metrics,
            default_metric=default_metric,
        )
