#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

import os
import pathlib
import subprocess
from io import StringIO

import ruamel.yaml as yaml

# Maps from environment variables to CVAR names.
# These are used to populate the C++ CVAR mappings.
# We populate these maps while iterating over the CVARs,
# and then we use them to generate the C++ code.
env_string_kv_pairs = {}
env_int64_kv_pairs = {}
env_int_kv_pairs = {}
env_bool_kv_pairs = {}

# Hard-coded CVARs for unit tests, so that changing the
# available CVARs does not risk breaking the unit tests.
string_cvar_for_unit_tests: str = "__NCCL_UNIT_TEST_STRING_CVAR__"
int64_cvar_for_unit_tests: str = "__NCCL_UNIT_TEST_INT64_T_CVAR__"
uint16_cvar_for_unit_tests: str = "__NCCL_UNIT_TEST_UINT16_T_CVAR__"
size_t_cvar_for_unit_tests: str = "__NCCL_UNIT_TEST_SIZE_T_CVAR__"
int_cvar_for_unit_tests: str = "__NCCL_UNIT_TEST_INT_CVAR__"
bool_cvar_for_unit_tests: str = "__NCCL_UNIT_TEST_BOOL_CVAR__"
double_cvar_for_unit_tests: str = "__NCCL_UNIT_TEST_DOUBLE_CVAR__"


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def numeric_type_to_kv_pair_map(type_name: str):
    """
    Returns the KV pair map name for a given numeric type.
    """
    if type_name == "int":
        return env_int_kv_pairs
    elif type_name == "int64_t":
        return env_int64_kv_pairs
    else:
        raise Exception(f"Unsupported numeric CVAR type '{type_name}'")


def numeric_type_to_map(type_name: str):
    """
    Returns the CVAR map name for a given numeric type.
    """
    if type_name == "int":
        return "env_int_values"
    elif type_name == "int64_t":
        return "env_int64_values"
    elif type_name == "double":
        return "env_double_values"
    elif type_name == "uint64_t":
        return "env_uint64_values"
    elif type_name == "size_t":
        return "env_size_t_values"
    elif type_name == "int32_t":
        return "env_int32_values"
    elif type_name == "uint32_t":
        return "env_uint32_values"
    elif type_name == "uint16_t":
        return "env_uint16_values"
    else:
        raise Exception(f"Unsupported numeric CVAR type '{type_name}'")


@static_vars(counter=0)
def indent(file, str_):
    str = str_.strip()
    if str[0] == "}":
        c = indent.counter - 1
    else:
        c = indent.counter
    spaces = "  " * c
    file.write("%s%s\n" % (spaces, str))
    indent.counter += str.count("{") - str.count("}")


class basetype:
    def __init__(self, cvar):
        self.name = cvar["name"]
        self.default = cvar["default"]
        self.description = cvar["description"]
        self.type = cvar["type"]
        if "envstr" in cvar:
            self.envstr = cvar["envstr"]
        else:
            self.envstr = self.name
        if "choices" in cvar:
            self.choices = cvar["choices"]
        else:
            self.choices = ""
        if "prefixes" in cvar:
            self.prefixes = cvar["prefixes"]
        else:
            self.prefixes = ""

    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        indent(file, "extern %s %s;" % (self.type, self.name))
        indent(file, "extern %s %s_DEFAULT;" % (self.type, self.name))
        file.write("\n")

    def storageDecl(self, file):
        indent(file, "%s %s;" % (self.type, self.name))
        indent(file, "%s %s_DEFAULT;" % (self.type, self.name))

    def desc(self, file):
        file.write("\n")
        if self.name == self.envstr:
            file.write("%s\n" % self.name)
        else:
            file.write(
                "%s (internal variable within NCCL: %s)\n" % (self.envstr, self.name)
            )
        file.write("Description:\n")
        d = self.description.split("\n")
        for line in d:
            file.write("    %s\n" % line)
        file.write("Type: %s\n" % self.type)
        file.write("Default: %s\n" % self.default)


class bool(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def readenv(self, file):
        global env_bool_kv_pairs

        indent(
            file, '%s = env2bool("%s", "%s");' % (self.name, self.envstr, self.default)
        )
        indent(
            file,
            '%s_DEFAULT = env2bool("NCCL_ENV_DO_NOT_SET", "%s");'
            % (self.name, self.default),
        )

        env_bool_kv_pairs[self.envstr] = self.name

        file.write("\n")


class numeric(basetype):
    def __init__(self, cvar):
        super().__init__(cvar)

    @staticmethod
    def utilfns(file):
        pass

    def readenv(self, file):
        indent(
            file,
            '%s = env2num<%s>("%s", "%s");'
            % (self.name, self.type, self.envstr, self.default),
        )
        indent(
            file,
            '%s_DEFAULT = env2num<%s>("NCCL_ENV_DO_NOT_SET", "%s");'
            % (self.name, self.type, self.default),
        )

        if self.type in ["int", "int64_t"]:
            numeric_type_to_kv_pair_map(self.type)[self.envstr] = self.name

        file.write("\n")


class double(numeric):
    pass


class string(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        indent(file, "extern std::string %s;" % self.name)
        indent(file, "extern std::string %s_DEFAULT;" % self.name)
        file.write("\n")

    def storageDecl(self, file):
        indent(file, "std::string %s;" % self.name)
        indent(file, "std::string %s_DEFAULT;" % self.name)

    def readenv(self, file):
        default = self.default if self.default else ""
        indent(file, '%s = env2str("%s", "%s");' % (self.name, self.envstr, default))
        indent(
            file,
            '%s_DEFAULT = env2str("NCCL_ENV_DO_NOT_SET", "%s");' % (self.name, default),
        )
        env_string_kv_pairs[self.envstr] = self.name
        file.write("\n")


class stringlist(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        indent(file, "extern std::vector<std::string> %s;" % self.name)
        indent(file, "extern std::vector<std::string> %s_DEFAULT;" % self.name)
        file.write("\n")

    def storageDecl(self, file):
        indent(file, "std::vector<std::string> %s;" % self.name)
        indent(file, "std::vector<std::string> %s_DEFAULT;" % self.name)

    def readenv(self, file):
        default = self.default if self.default else ""
        indent(file, "%s.clear();" % self.name)
        indent(
            file, '%s = env2strlist("%s", "%s");' % (self.name, self.envstr, default)
        )
        indent(file, "%s_DEFAULT.clear();" % self.name)
        indent(
            file,
            '%s_DEFAULT = env2strlist("NCCL_ENV_DO_NOT_SET", "%s");'
            % (self.name, default),
        )
        file.write("\n")


class dictlist(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        indent(
            file,
            "extern std::unordered_map<std::string,std::vector<std::string>> %s;"
            % self.name,
        )
        indent(
            file,
            "extern std::unordered_map<std::string,std::vector<std::string>> %s_DEFAULT;"
            % self.name,
        )
        file.write("\n")

    def storageDecl(self, file):
        indent(
            file,
            "std::unordered_map<std::string,std::vector<std::string>> %s;" % self.name,
        )
        indent(
            file,
            "std::unordered_map<std::string,std::vector<std::string>> %s_DEFAULT;"
            % self.name,
        )

    def readenv(self, file):
        default = self.default if self.default else ""
        indent(file, "%s.clear();" % self.name)
        indent(
            file, '%s = env2dictlist("%s", "%s");' % (self.name, self.envstr, default)
        )
        indent(file, "%s_DEFAULT.clear();" % self.name)
        indent(
            file,
            '%s_DEFAULT = env2dictlist("NCCL_ENV_DO_NOT_SET", "%s");'
            % (self.name, default),
        )
        file.write("\n")


class prefixedStringlist(stringlist):
    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        indent(file, "extern std::string %s_PREFIX;" % self.name)
        indent(file, "extern std::string %s_PREFIX_DEFAULT;" % self.name)
        super().externDecl(file)

    def storageDecl(self, file):
        indent(file, "std::string %s_PREFIX;" % self.name)
        indent(file, "std::string %s_PREFIX_DEFAULT;" % self.name)
        super().storageDecl(file)

    def readenv(self, file):
        trimmedPrefixes = [v.strip() for v in self.prefixes.split(",")]
        indent(
            file,
            'std::vector<std::string> %s_allPrefixes{"%s"};'
            % (self.name, ('", "').join(trimmedPrefixes)),
        )
        default = self.default if self.default else ""
        indent(file, "%s.clear();" % self.name)
        indent(
            file,
            'std::tie(%s_PREFIX, %s) = env2prefixedStrlist("%s", "%s", %s_allPrefixes);'
            % (self.name, self.name, self.envstr, default, self.name),
        )
        indent(file, "%s_DEFAULT.clear();" % self.name)
        indent(
            file,
            "std::tie(%s_PREFIX_DEFAULT, %s_DEFAULT) = "
            'env2prefixedStrlist("NCCL_ENV_DO_NOT_SET", "%s", %s_allPrefixes);'
            % (self.name, self.name, default, self.name),
        )
        file.write("\n")


class enum(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        choiceList = self.choices.replace(" ", "").split(",")
        indent(file, "enum class %s {" % self.name)
        for c in choiceList:
            indent(file, "%s," % c)
        indent(file, "};")
        indent(file, "extern enum %s %s;" % (self.name, self.name))
        indent(file, "extern enum %s %s_DEFAULT;" % (self.name, self.name))
        file.write("\n")

    def storageDecl(self, file):
        indent(file, "enum %s %s;" % (self.name, self.name))
        indent(file, "enum %s %s_DEFAULT;" % (self.name, self.name))

    def readenv(self, file):
        indent(file, 'if (getenv("%s") == nullptr) {' % self.envstr)
        indent(file, "%s = %s::%s;" % (self.name, self.name, self.default))
        indent(file, "} else {")
        indent(file, 'std::string str(getenv("%s"));' % self.envstr)
        choices = self.choices.replace(" ", "").split(",")
        for idx, c in enumerate(choices):
            if idx == 0:
                indent(file, 'if (str == std::string("%s")) {' % c)
            else:
                indent(file, '} else if (str == std::string("%s")) {' % c)
            indent(file, "%s = %s::%s;" % (self.name, self.name, c))
        indent(file, "} else {")
        indent(file, '  CVAR_WARN_UNKNOWN_VALUE("%s", str.c_str());' % self.name)
        indent(file, "}")
        indent(file, "}")
        indent(file, "%s_DEFAULT = %s::%s;" % (self.name, self.name, self.default))
        file.write("\n")


class enumlist(basetype):
    @staticmethod
    def utilfns(file):
        pass

    def externDecl(self, file):
        choiceList = self.choices.replace(" ", "").split(",")
        indent(file, "enum class %s {" % self.name)
        for c in choiceList:
            indent(file, "%s," % c)
        indent(file, "};")
        indent(file, "extern std::vector<enum %s> %s;" % (self.name, self.name))
        indent(file, "extern std::vector<enum %s> %s_DEFAULT;" % (self.name, self.name))
        file.write("\n")

    def storageDecl(self, file):
        indent(file, "std::vector<enum %s> %s;" % (self.name, self.name))
        indent(file, "std::vector<enum %s> %s_DEFAULT;" % (self.name, self.name))

    def readenv(self, file):
        indent(file, "{")
        indent(file, "%s.clear();" % self.name)
        indent(
            file, 'auto tokens = env2strlist("%s", "%s");' % (self.envstr, self.default)
        )
        choices = self.choices.replace(" ", "").split(",")
        indent(file, "for (auto token : tokens) {")
        for idx, c in enumerate(choices):
            if idx == 0:
                indent(file, 'if (token == std::string("%s")) {' % c)
            else:
                indent(file, '} else if (token == std::string("%s")) {' % c)
            indent(file, "%s.emplace_back(%s::%s);" % (self.name, self.name, c))
        indent(file, "} else {")
        indent(file, '  CVAR_WARN_UNKNOWN_VALUE("%s", token.c_str());' % self.name)
        indent(file, "}")
        indent(file, "}")
        indent(file, "}")
        indent(file, "%s_DEFAULT.clear();" % self.name)
        default = self.default.replace(" ", "").split(",")
        for d in default:
            indent(file, "%s_DEFAULT.emplace_back(%s::%s);" % (self.name, self.name, d))
        file.write("\n")


def sign_source(file):
    """
    Adds a comment with a SignedSource using current hg commit id as a token.
    """
    hgId = subprocess.check_output(["hg", "whereami"]).decode("utf-8").strip()
    file.write("// @" + "generated SignedSource<<%s>>\n" % hgId)


def printAutogenStart(file):
    file.write(
        "// Automatically generated by ./comms/utils/cvars/extractcvars.py --- START\n"
    )
    file.write("// DO NOT EDIT!!!\n\n")


def printAutogenHeader(file):
    file.write("// Copyright (c) Meta Platforms, Inc. and affiliates.\n")
    sign_source(file)
    printAutogenStart(file)


def printAutogenFooter(file):
    file.write(
        "// Automatically generated by ./comms/utils/cvars/extractcvars.py --- END\n"
    )


def populateValidator(file):
    # Generate cvars validator to make sure no cvar conflicts
    indent(file, "static void validateCvarEnv() {")
    indent(
        file,
        'if (NCCL_SOCKET_IFNAME.find("beth") != NCCL_CLIENT_SOCKET_IFNAME.find("beth")) {',
    )
    indent(
        file,
        'CVAR_ERROR( "CVAR incompatible: NCCL_SOCKET_IFNAME({}) vs NCCL_CLIENT_SOCKET_IFNAME({})", NCCL_SOCKET_IFNAME, NCCL_CLIENT_SOCKET_IFNAME);',
    )
    indent(file, "}")
    indent(file, "}")
    file.write("\n")


def declareCvarMaps(file):
    """
    Declare the maps used to store the cvar values.

    Arguments:
        file: file to write to
    """
    cvar_maps_text: str = """
std::unordered_map<std::string, std::string*> env_string_values = {<@env_string_kv_pairs>};
std::unordered_map<std::string, int64_t*> env_int64_values = {<@env_int64_kv_pairs>};
std::unordered_map<std::string, int*> env_int_values = {<@env_int_kv_pairs>};
std::unordered_map<std::string, bool*> env_bool_values = {<@env_bool_kv_pairs>};

    """

    indent(file, cvar_maps_text)


def updateCvarMapDeclarations(fileContents: str) -> str:
    maps: dict[str, dict[str, str]] = {
        "env_string_kv_pairs": env_string_kv_pairs,
        "env_int64_kv_pairs": env_int64_kv_pairs,
        "env_int_kv_pairs": env_int_kv_pairs,
        "env_bool_kv_pairs": env_bool_kv_pairs,
    }

    for str_to_replace, cvar_map in maps.items():
        map_declaration: str = ""

        for envstr, cvar_name in cvar_map.items():
            map_declaration += f'{{"{envstr}", &{cvar_name}}},\n'

        fileContents = fileContents.replace(f"<@{str_to_replace}>", map_declaration)

    return fileContents


def populateCCFile(
    allcvars,
    templateFilename: str,
    outputFilename: str,
):
    """
    Generate the CVARS C++ file.

    Arguments:
        allcvars: list of cvars.
        templateFilename: template file name.
        outputFilename: output file name.
    """
    file = StringIO()

    # Generate storage declaration
    for cvar in allcvars:
        cvar.storageDecl(file)

    file.write("\n")

    # Generate initialization for environment variable set and maps
    indent(file, "namespace ncclx {")

    declareCvarMaps(file)

    file.write("\n")
    indent(file, "static void initEnvSet(std::unordered_set<std::string>& env) {")
    for cvar in allcvars:
        indent(file, 'env.insert("%s");' % cvar.envstr)

    indent(file, "}")
    file.write("\n")

    populateValidator(file)

    # Generate environment reading of all cvars
    indent(file, "static void readCvarEnv() {")
    for cvar in allcvars:
        cvar.readenv(file)
        indent(file, f'if ({cvar.name}_DEFAULT != {cvar.name}) {"{"}')
        indent(
            file,
            f'  CVAR_INFO("NCCL Config - CVAR {{}} has an override", "{cvar.name}");',
        )
        indent(file, "}")

    indent(file, "ncclx::validateCvarEnv();")
    indent(file, "}")
    indent(file, "};")
    file.write("\n")

    content = file.getvalue()

    # Load template and insert generated contents
    with open(templateFilename, "r") as tpl:
        fileContents = tpl.read()
        fileContents = fileContents.replace("### AUTOGEN_CONTENT ###", content)
        fileContents = updateCvarMapDeclarations(fileContents)
        with open(outputFilename, "w") as out:
            printAutogenHeader(out)
            out.write(fileContents)
            printAutogenFooter(out)

    file.close()


def populateHFile(allcvars, outputFilename):
    """
    Generate the CVARS header file.

    Arguments:
        allcvars: list of cvars
        outputFilename: output file name
    """
    file = open(outputFilename, "w")
    printAutogenHeader(file)
    file.write("\n")

    file.write("#ifndef NCCL_CVARS_H_INCLUDED\n")
    file.write("#define NCCL_CVARS_H_INCLUDED\n")
    file.write("\n")

    file.write("#include <cstdint>\n")
    file.write("#include <string>\n")
    file.write("#include <vector>\n")
    file.write("#include <unordered_map>\n")
    file.write("\n")

    # Generate extern declaration
    for cvar in allcvars:
        cvar.externDecl(file)
    file.write("\n")
    file.write("namespace ncclx {\n")

    cvar_maps_text: str = """
extern std::unordered_map<std::string, std::string*> env_string_values;
extern std::unordered_map<std::string, int64_t*> env_int64_values;
extern std::unordered_map<std::string, int*> env_int_values;
extern std::unordered_map<std::string, bool*> env_bool_values;

"""

    file.write(cvar_maps_text)

    file.write("};\n")
    file.write("void ncclCvarInit();\n")
    file.write("\n")

    file.write("#endif  /* NCCL_CVARS_H_INCLUDED */")
    file.write("\n")

    printAutogenFooter(file)
    file.close()


def format_file(path):
    subprocess.run(["clang-format", "-i", path])


def append_unit_test_cvars(allcvars: list) -> None:
    allcvars.append(
        string(
            {
                "name": string_cvar_for_unit_tests,
                "type": "string",
                "default": "",
                "description": "string-typed CVAR for use in unit tests",
            }
        )
    )

    allcvars.append(
        bool(
            {
                "name": bool_cvar_for_unit_tests,
                "type": "bool",
                "default": False,
                "description": "bool-typed CVAR for use in unit tests",
            }
        )
    )

    allcvars.append(
        numeric(
            {
                "name": int_cvar_for_unit_tests,
                "type": "int",
                "default": 0,
                "description": "int-typed CVAR for use in unit tests",
            }
        )
    )

    allcvars.append(
        numeric(
            {
                "name": int64_cvar_for_unit_tests,
                "type": "int64_t",
                "default": 0,
                "description": "int64_t-typed CVAR for use in unit tests",
            }
        )
    )

    allcvars.append(
        numeric(
            {
                "name": uint16_cvar_for_unit_tests,
                "type": "uint16_t",
                "default": 0,
                "description": "uint16_t-typed CVAR for use in unit tests",
            }
        )
    )

    allcvars.append(
        numeric(
            {
                "name": size_t_cvar_for_unit_tests,
                "type": "size_t",
                "default": 0,
                "description": "size_t-typed CVAR for use in unit tests",
            }
        )
    )

    allcvars.append(
        numeric(
            {
                "name": double_cvar_for_unit_tests,
                "type": "double",
                "default": 0,
                "description": "double-typed CVAR for use in unit tests",
            }
        )
    )


def main():
    # Get the directory of the current script
    script_dir = pathlib.Path(__file__).parent
    config_file = os.path.join(script_dir, "nccl_cvars.yaml")

    print(f"Parsing NCCL env variables from {config_file}")
    with open(config_file, "r") as f:
        data = yaml.YAML().load(f.read())
    if data["cvars"] is None:
        data["cvars"] = []

    loadedCvars = sorted(data["cvars"], key=lambda x: x["name"])

    allcvars = []
    for cvar in loadedCvars:
        if cvar["type"] == "bool":
            allcvars.append(bool(cvar))
        elif cvar["type"] == "string":
            allcvars.append(string(cvar))
        elif cvar["type"] == "stringlist":
            allcvars.append(stringlist(cvar))
        elif cvar["type"] == "enum":
            allcvars.append(enum(cvar))
        elif cvar["type"] == "enumlist":
            allcvars.append(enumlist(cvar))
        elif cvar["type"] == "prefixed_stringlist":
            allcvars.append(prefixedStringlist(cvar))
        elif cvar["type"] == "double":
            allcvars.append(double(cvar))
        elif cvar["type"] == "dictlist":
            allcvars.append(dictlist(cvar))
        else:
            allcvars.append(numeric(cvar))

    append_unit_test_cvars(allcvars)

    populateCCFile(
        allcvars,
        "comms/utils/cvars/nccl_cvars.cc.in",
        "comms/utils/cvars/nccl_cvars.cc",
    )
    populateHFile(allcvars, "comms/utils/cvars/nccl_cvars.h")
    for f in [
        "comms/utils/cvars/nccl_cvars.cc",
        "comms/utils/cvars/nccl_cvars.h",
    ]:
        format_file(f)


if __name__ == "__main__":
    main()
