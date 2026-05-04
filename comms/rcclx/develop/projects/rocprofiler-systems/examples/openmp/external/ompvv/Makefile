SHELL=/bin/bash -o pipefail

# When using make alone show the help message
.DEFAULT_GOAL:=help

###################################################
# OpenMP Versions. This is to support multiple
# versions of the standard in the same testsuite
###################################################
OMP_VERSION?=4.5
TEST_FOLDER_EXISTS=$(shell if [ -d tests/$(OMP_VERSION) ]; then echo "exist";fi)
ifeq "$(TEST_FOLDER_EXISTS)" ""
  $(error The OMP_VERSION=$(OMP_VERSION) does not exist in this testsuite)
endif


###################################################
# System specific varibles can be specified
# in the system files sys/system/###.def
###################################################
ifdef OMPVV_SYSTEM
	SYSTEM = ${OMPVV_SYSTEM}
else
	SYSTEM ?= generic
endif
-include sys/systems/$(SYSTEM).def

# Set OMPVV_NO_COMPILER_MODULE_CHANGES to prevent
# changes to the compiler settings inside make.def
include sys/make/make.def

###################################################
# It is possible to annotate a set of results with
# a LOG_NOTE which will be added to the log header
# files
###################################################
LOG_NOTE?="none"

###################################################
# Verbose & Log
###################################################
QUIET:=@
ifdef VERBOSE
  QUIET:=
endif

RECORD:=
LOGDIR:=
LOGDIRNAME ?= logs
ifdef LOG
  LOGDIR:= $(LOGDIRNAME)
  RECORD:= | tee -a $(LOGDIR)/
endif
ifdef LOG_ALL
  LOG:=1 # LOG_ALL implies LOG
  LOGDIR:= $(LOGDIRNAME)
  RECORD:= 2>&1 | tee -a $(LOGDIR)/
endif

ifdef VERBOSE_TESTS
  VERBOSE_MODE = -DVERBOSE_MODE=1
endif

BSRUN:=
ifdef ADD_BATCH_SCHED
  BSRUN:= $(BATCH_SCHEDULER)
endif

# Test running and results analyzer
RUN_TEST=$(CURDIR)/sys/scripts/run_test.sh
RESULTS_ANALYZER=$(CURDIR)/sys/scripts/createSummary.py
RESULTS_JSON_OUTPUT_FILE=results.json
RESULTS_CSV_OUTPUT_FILE=results.csv
RESULTS_HTML_OUTPUT_FOLDER=results_report
RESULTS_HTML_REPORT_TEMPLATE=$(CURDIR)/sys/results_template
REPORT_ONLINE_CONNECTION=$(CURDIR)/sys/scripts/onlineConnection.py

##################################################
# Source files
##################################################

ifneq "$(SOURCES_C)$(SOURCES_CPP)$(SOURCES_F)$(TESTS_TO_RUN)" ""
$(error The SOURCES_C SOURCES_CPP SOURCES_F and TESTS_TO_RUN flags were depreciated. Use SOURCES instead)
endif

ifneq "$(SOURCES)" ""
# Obtain all the possible source files for C, CPP and Fortran
SOURCES_C := $(shell find $(CURDIR)/tests/$(OMP_VERSION) -path "*$(SOURCES)" | grep "\.c$$")
SOURCES_CPP := $(shell find $(CURDIR)/tests/$(OMP_VERSION) -path "*$(SOURCES)" | grep "\.cpp$$")
SOURCES_F := $(shell find $(CURDIR)/tests/$(OMP_VERSION) -path "*$(SOURCES)" | grep "\(\.F90\|\.F95\|\.F03\|\.F\|\.FOR\)$$")
$(info SOURCES = $(notdir $(SOURCES_C) $(SOURCES_CPP) $(SOURCES_F)))

# Obtain the list of files that were previously
# compiled based on the subsection of tests
TESTS_TO_RUN := $(foreach testName, \
                  $(notdir $(SOURCES_C) $(SOURCES_CPP) $(SOURCES_F)), \
                  $(shell test -d $(BINDIR) && find $(BINDIR) -name "$(testName)*"))
TESTS_TO_RUN := $(TESTS_TO_RUN:.FOR.o=.FOR.FOR.o) # Adding .FOR.o to fortran
TESTS_TO_RUN := $(TESTS_TO_RUN:.F90.o=.F90.FOR.o)
TESTS_TO_RUN := $(TESTS_TO_RUN:.F95.o=.F95.FOR.o)
TESTS_TO_RUN := $(TESTS_TO_RUN:.F03.o=.F03.FOR.o)
TESTS_TO_RUN := $(TESTS_TO_RUN:.F.o=.F.FOR.o)
RUN_TESTS = $(TESTS_TO_RUN:.o=.runonly)

# Obtain all the possible binary files formed from the source files
ifneq "$(CC)" "none"
OBJS_C := $(SOURCES_C:.c=.c.o)
endif
ifneq "$(CXX)" "none"
OBJS_CPP := $(SOURCES_CPP:.cpp=.cpp.o)
endif
ifneq "$(FC)" "none"
OBJS_F := $(SOURCES_F:.FOR=.FOR.FOR.o)
OBJS_F := $(OBJS_F:.F90=.F90.FOR.o)
OBJS_F := $(OBJS_F:.F95=.F95.FOR.o)
OBJS_F := $(OBJS_F:.F03=.F03.FOR.o)
OBJS_F := $(OBJS_F:.F=.F.FOR.o)
endif

# Build list of compilation dependencies
COMP_DEP = $(OBJS_C) $(OBJS_CPP) $(OBJS_F)

# Obtain the list of dependencies for the rule all
ALL_DEP :=
ifneq "$(CC)" "none"
ALL_DEP += $(addprefix $(BINDIR)/,$(notdir $(SOURCES_C:.c=.c.run)))
endif
ifneq "$(CXX)" "none"
ALL_DEP += $(addprefix $(BINDIR)/,$(notdir $(SOURCES_CPP:.cpp=.cpp.run)))
endif
ifneq "$(FC)" "none"
ALL_DEP += $(addprefix $(BINDIR)/,$(notdir $(SOURCES_F:.F90=.F90.FOR.run)))
ALL_DEP := $(addprefix $(BINDIR)/,$(notdir $(ALL_DEP:.F95=.F95.FOR.run)))
ALL_DEP := $(addprefix $(BINDIR)/,$(notdir $(ALL_DEP:.F03=.F03.FOR.run)))
ALL_DEP := $(addprefix $(BINDIR)/,$(notdir $(ALL_DEP:.F=.F.FOR.run)))
ALL_DEP := $(addprefix $(BINDIR)/,$(notdir $(ALL_DEP:.FOR=.FOR.FOR.run)))
endif

endif

ifeq "$(SOURCES)" ""
# Getting all the source files in the project
ifdef LINK_OMPVV_LIB
SOURCES_C := $(shell find $(CURDIR)/tests/$(OMP_VERSION) -name "*.c")
else
SOURCES_C := $(shell find $(CURDIR)/tests/$(OMP_VERSION) ! -name qmcpack_target_static_lib.c -name "*.c")
endif
SOURCES_CPP := $(shell find $(CURDIR)/tests/$(OMP_VERSION) -name "*.cpp")
SOURCES_F := $(shell find $(CURDIR)/tests/$(OMP_VERSION) -name "*.F90" -o -name "*.F95" -o -name "*.F03" -o -name "*.F" -o -name "*.FOR" | grep -v "ompvv.F90")

# Find all the binary files that have been previously compiled
TESTS_TO_RUN := $(shell test -d $(BINDIR) && \
                        find $(BINDIR) -name "*.F90.o" \
                        -o -name "*.F95.o" \
                        -o -name "*.F03.o" \
                        -o -name "*.F.o" \
                        -o -name "*.FOR.o" \
                        -o -name "*.c.o" \
                        -o -name "*.cpp.o")
TESTS_TO_RUN := $(TESTS_TO_RUN:.FOR.o=.FOR.FOR.o) # Adding .FOR.o to fortran
TESTS_TO_RUN := $(TESTS_TO_RUN:.F90.o=.F90.FOR.o)
TESTS_TO_RUN := $(TESTS_TO_RUN:.F95.o=.F95.FOR.o)
TESTS_TO_RUN := $(TESTS_TO_RUN:.F03.o=.F03.FOR.o)
TESTS_TO_RUN := $(TESTS_TO_RUN:.F.o=.F.FOR.o)
RUN_TESTS := $(TESTS_TO_RUN:.o=.runonly)

# Creating compile dependencies
ifneq "$(CC)" "none"
OBJS_C := $(SOURCES_C:.c=.c.o)
endif
ifneq "$(CXX)" "none"
OBJS_CPP := $(SOURCES_CPP:.cpp=.cpp.o)
endif
ifneq "$(FC)" "none"
OBJS_F := $(SOURCES_F:.F90=.F90.FOR.o)
OBJS_F := $(OBJS_F:.F95=.F95.FOR.o)
OBJS_F := $(OBJS_F:.F03=.F03.FOR.o)
OBJS_F := $(OBJS_F:.F=.F.FOR.o)
OBJS_F := $(OBJS_F:.FOR=.FOR.FOR.o)
endif
COMP_DEP := $(OBJS_C) $(OBJS_CPP) $(OBJS_F)

# Get all the dependencies for all rule
ALL_DEP :=
ifneq "$(CC)" "none"
ALL_DEP := $(addprefix $(BINDIR)/,$(notdir $(SOURCES_C:.c=.c.run)))
endif
ifneq "$(CXX)" "none"
ALL_DEP += $(addprefix $(BINDIR)/,$(notdir $(SOURCES_CPP:.cpp=.cpp.run)))
endif
ifneq "$(FC)" "none"
ALL_DEP += $(addprefix $(BINDIR)/,$(notdir $(SOURCES_F:.F90=.F90.FOR.run)))
ALL_DEP := $(addprefix $(BINDIR)/,$(notdir $(ALL_DEP:.F95=.F95.FOR.run)))
ALL_DEP := $(addprefix $(BINDIR)/,$(notdir $(ALL_DEP:.F03=.F03.FOR.run)))
ALL_DEP := $(addprefix $(BINDIR)/,$(notdir $(ALL_DEP:.F=.F.FOR.run)))
ALL_DEP := $(addprefix $(BINDIR)/,$(notdir $(ALL_DEP:.FOR=.FOR.FOR.run)))
endif

endif

$(ALL_DEP) $(COMP_DEP) $(RUN_TESTS): LOGTEMPFILE := $(LOGTEMPFILE)

.PHONY: all
all: MessageDisplay $(ALL_DEP)
	@echo "====COMPILE AND RUN DONE===="


.PHONY: compile
compile: MessageDisplay $(COMP_DEP)
	@echo "====COMPILE DONE===="

##################################################
# FOR RUNNING TESTS ONLY
##################################################

.PHONY: run
run: $(RUN_TESTS)
	@echo "====RUN DONE====="

.PHONY: MessageDisplay
MessageDisplay:
	@echo "    ====    SOLLVE PROJECT MAKEFILE   ====   "
	@echo "Running make with the following compilers"
ifneq "$(CC)" "none"
	@echo "CC = $(CC) $(shell $(call loadModules,$(C_COMPILER_MODULE),"shut up") ${C_VERSION})"
endif
ifneq "$(CXX)" "none"
	@echo "CXX = $(CXX) $(shell $(call loadModules,$(CXX_COMPILER_MODULE),"shut up") ${CXX_VERSION})"
endif
ifneq "$(FC)" "none"
	@echo "FC = $(FC) $(shell $(call loadModules,$(F_COMPILER_MODULE),"shut up") ${F_VERSION})"
endif
	$(if $(MODULE_LOAD), @echo "C_MODULE = "$(subst ;,\;,$(C_COMPILER_MODULE)); echo "CXX_MODULE = "$(subst ;,\;,${CXX_COMPILER_MODULE}); echo "F_MODULE = "$(subst ;,\;,${F_COMPILER_MODULE}),)

##################################################
# Turn off offloading
##################################################

ifdef NO_OFFLOADING
  COFFLOADING = $(C_NO_OFFLOADING)
  CXXOFFLOADING = $(CXX_NO_OFFLOADING)
  FOFFLOADING = $(F_NO_OFFLOADING)
endif

##################################################
# OMPVV static library
##################################################

OMPVVLIB=-L$(CURDIR)/ompvv -lompvv
OMPVVLIB_DEP=$(CURDIR)/ompvv/libompvv.a

$(BINDIR)/libompvv.o: $(CURDIR)/ompvv/libompvv.c $(BINDIR)
	@echo -e $(TXTYLW)"\n\n" compile: $< $(TXTNOC)
	$(call log_section_header,"COMPILE CC="${CCOMPILE},$(SYSTEM),$<,$(CC) $(shell $(call loadModules,$(C_COMPILER_MODULE),"shut up") $(C_VERSION)),$(OMP_VERSION),$(notdir $(@:.o=.log)))
	-$(QUIET)$(call loadModules,$(C_COMPILER_MODULE)) $(CCOMPILE) $(VERBOSE_MODE) $(DTHREADS) $(DTEAMS) $(HTHREADS) $< -c -o $(BINDIR)/$(notdir $@) $(if $(LOG),$(RECORD)$(notdir $(@:.o=.log))\
		&& echo "PASS" > $(LOGTEMPFILE) \
		|| echo "FAIL" > $(LOGTEMPFILE))
	-$(call log_section_footer,"COMPILE CC="${CCOMPILE},$(SYSTEM),$$(cat $(LOGTEMPFILE)),$(LOG_NOTE),$(notdir $(@:.o=.log)))
	-@$(if $(LOG), rm $(LOGTEMPFILE))

$(CURDIR)/ompvv/libompvv.a: $(BINDIR)/libompvv.o
	ar -rc $(CURDIR)/ompvv/libompvv.a $(BINDIR)/libompvv.o
	@ranlib $(CURDIR)/ompvv/libompvv.a

##################################################
# Compilation rules
##################################################
# c files rule
%.c.o: %.c $(BINDIR) $(LOGDIR)
	@echo -e $(TXTYLW)"\n\n" compile: $< $(TXTNOC)
	$(call log_section_header,"COMPILE CC="${CCOMPILE},$(SYSTEM),$<,$(CC) $(shell $(call loadModules,$(C_COMPILER_MODULE),"shut up") $(C_VERSION)),$(OMP_VERSION),$(notdir $(@:.o=.log)))
	-$(QUIET)$(call loadModules,$(C_COMPILER_MODULE)) $(CCOMPILE) $(VERBOSE_MODE) $(DTHREADS) $(DTEAMS) $(HTHREADS) $< -o $(BINDIR)/$(notdir $@) $(if $(LOG),$(RECORD)$(notdir $(@:.o=.log))\
		&& echo "PASS" > $(LOGTEMPFILE) \
		|| echo "FAIL" > $(LOGTEMPFILE))
	-$(call log_section_footer,"COMPILE CC="${CCOMPILE},$(SYSTEM),$$(cat $(LOGTEMPFILE)),$(LOG_NOTE),$(notdir $(@:.o=.log)))
	-@$(if $(LOG), rm $(LOGTEMPFILE))

# Special rule for test that needs OMPVV lib
$(CURDIR)/tests/4.5/application_kernels/qmcpack_target_static_lib.c.o: $(CURDIR)/tests/4.5/application_kernels/qmcpack_target_static_lib.c $(BINDIR) $(LOGDIR) $(OMPVVLIB_DEP)
	@echo -e $(TXTYLW)"\n\n" compile: $< $(TXTNOC)
	$(call log_section_header,"COMPILE CC="${CCOMPILE},$(SYSTEM),$<,$(CC) $(shell $(call loadModules,$(C_COMPILER_MODULE),"shut up") $(C_VERSION)),$(OMP_VERSION),$(notdir $(@:.o=.log)))
	-$(QUIET)$(call loadModules,$(C_COMPILER_MODULE)) $(CCOMPILE) $(VERBOSE_MODE) $(DTHREADS) $(DTEAMS) $(HTHREADS) $< -o $(BINDIR)/$(notdir $@) $(OMPVVLIB) $(if $(LOG),$(RECORD)$(notdir $(@:.o=.log))\
		&& echo "PASS" > $(LOGTEMPFILE) \
		|| echo "FAIL" > $(LOGTEMPFILE))
	-$(call log_section_footer,"COMPILE CC="${CCOMPILE},$(SYSTEM),$$(cat $(LOGTEMPFILE)),$(LOG_NOTE),$(notdir $(@:.o=.log)))
	-@$(if $(LOG), rm $(LOGTEMPFILE))

# c++ files rule
%.cpp.o: %.cpp $(BINDIR) $(LOGDIR)
	@echo -e $(TXTYLW)"\n\n" compile: $< $(TXTNOC)
	$(call log_section_header,"COMPILE CPP="${CXXCOMPILE},$(SYSTEM),$<,$(CXX) $(shell $(call loadModules,$(CXX_COMPILER_MODULE),"shut up") $(CXX_VERSION)),$(OMP_VERSION),$(notdir $(@:.o=.log)))
	-$(QUIET)$(call loadModules,$(CXX_COMPILER_MODULE)) $(CXXCOMPILE) $(VERBOSE_MODE) $(DTHREADS) $(DTEAMS) $(HTHREADS) $< -o $(BINDIR)/$(notdir $@) $(if $(LOG),$(RECORD)$(notdir $(@:.o=.log))\
		&& echo "PASS" > $(LOGTEMPFILE) \
		|| echo "FAIL" > $(LOGTEMPFILE))
	-$(call log_section_footer,"COMPILE",$(SYSTEM),$$(cat $(LOGTEMPFILE)),$(LOG_NOTE),$(notdir $(@:.o=.log)))
	-@$(if $(LOG), rm $(LOGTEMPFILE))

# fortran files rule
%.FOR.o: % $(BINDIR) $(LOGDIR) clear_fortran_mod
	@echo -e $(TXTYLW)"\n\n" compile: $< $(TXTNOC)
	$(call log_section_header,"COMPILE F="${FCOMPILE},$(SYSTEM),$<,$(FC) $(shell $(call loadModules,$(F_COMPILER_MODULE),"shut up") $(F_VERSION)),$(OMP_VERSION),$(notdir $(@:.FOR.o=.log)))
	-$(QUIET)$(call loadModules,$(F_COMPILER_MODULE)) $(FCOMPILE) $(VERBOSE_MODE) $(DTHREADS) $(DTEAMS) $(HTHREADS) $< -o $(BINDIR)/$(notdir $(@:.FOR.o=.o)) $(if $(LOG),$(RECORD)$(notdir $(@:.FOR.o=.log))\
		&& echo "PASS" > $(LOGTEMPFILE) \
		|| echo "FAIL" > $(LOGTEMPFILE))
	-$(call log_section_footer,"COMPILE",$(SYSTEM),$$(cat $(LOGTEMPFILE)),$(LOG_NOTE),$(notdir $(@:.FOR.o=.log)))
	-@$(if $(LOG), rm $(LOGTEMPFILE))


##################################################
# Running tests rules
##################################################
# run c app rule
%.c.run: $(OBJS_C)
	$(call log_section_header,"RUN",$(SYSTEM),$(@:.run=),$(LOG_NOTE),$(OMP_VERSION),$(notdir $(@:.run=.log)))
	@echo -e $(TXTGRN)"\n\n" running: $@ $(TXTNOC) $(if $(LOG), ${RECORD}$(notdir $(@:.run=.log)))

# If .../test_<envname>_env_<value>...
	$(if $(findstring _THREADS_RESERVE,$@), \
  		-$(call loadModules,$(C_COMPILER_MODULE)) \
   			$(BSRUN)$(RUN_TEST) --env OMP_THREAD_LIMIT 16 \
				--env OMP_THREADS_RESERVE "structured(5)$(comma)free_agent(2)" \
     			$(@:.run=.o) $(VERBOSE) $(if $(LOG),$(RECORD)$(notdir $(@:.run=.log)) \
     		&& echo "PASS" > $(LOGTEMPFILE) || echo "FAIL" > $(LOGTEMPFILE)), \
	$(if $(findstring _env_,$@), \
	  -$(call loadModules,$(C_COMPILER_MODULE)) \
		 $(BSRUN)$(RUN_TEST) --env \
		   $(shell echo "$@" | sed -e 's@.*/@@' -e 's@test_\(.*\)_env_.*@\1@' | tr 'a-z' 'A-Z') \
		   $(shell echo "$@" | sed -e 's@.*/@@' -e 's@.*_env_\([^.]*\).*@\1@') \
		   $(@:.run=.o) $(VERBOSE) $(if $(LOG),$(RECORD)$(notdir $(@:.run=.log)) \
		 && echo "PASS" > $(LOGTEMPFILE) \
		 || echo "FAIL" > $(LOGTEMPFILE)), \
	 -$(call loadModules,$(C_COMPILER_MODULE)) $(BSRUN)$(RUN_TEST) $(@:.run=.o) $(VERBOSE) $(if $(LOG),$(RECORD)$(notdir $(@:.run=.log))\
		 && echo "PASS" > $(LOGTEMPFILE) \
		 || echo "FAIL" > $(LOGTEMPFILE)) \
	))
	-$(call log_section_footer,"RUN",$(SYSTEM),$$(cat $(LOGTEMPFILE)),$(LOG_NOTE),$(notdir $(@:.run=.log)))
	-@$(if $(LOG), rm $(LOGTEMPFILE))


# run c++ app rule
%.cpp.run: $(OBJS_CPP)
	$(call log_section_header,"RUN",$(SYSTEM),$(@:.run=),$(LOG_NOTE),$(OMP_VERSION),$(notdir $(@:.run=.log)))
	@echo -e $(TXTGRN)"\n\n" running: $@ $(TXTNOC) $(if $(LOG), ${RECORD}$(notdir $(@:.run=.log)))
	-$(call loadModules,$(CXX_COMPILER_MODULE)) $(BSRUN)$(RUN_TEST) $(@:.run=.o) $(VERBOSE) $(if $(LOG),$(RECORD)$(notdir $(@:.run=.log))\
		&& echo "PASS" > $(LOGTEMPFILE) \
		|| echo "FAIL" > $(LOGTEMPFILE))
	-$(call log_section_footer,"RUN",$(SYSTEM),$$(cat $(LOGTEMPFILE)),$(LOG_NOTE),$(notdir $(@:.run=.log)))
	-@$(if $(LOG), rm $(LOGTEMPFILE))

# run Fortran app rule
%.FOR.run: $(OBJS_F)
	$(call log_section_header,"RUN",$(SYSTEM),$(@:.FOR.run=),$(LOG_NOTE),$(OMP_VERSION),$(notdir $(@:.FOR.run=.log)))
	@echo -e $(TXTGRN)"\n\n" running: $@ $(TXTNOC) $(if $(LOG), ${RECORD}$(notdir $(@:.FOR.run=.log)))
	-$(call loadModules,$(F_COMPILER_MODULE)) $(BSRUN)$(RUN_TEST) $(@:.FOR.run=.o) $(VERBOSE) $(if $(LOG),$(RECORD)$(notdir $(@:.FOR.run=.log))\
		&& echo "PASS" > $(LOGTEMPFILE) \
		|| echo "FAIL" > $(LOGTEMPFILE))
	-$(call log_section_footer,"RUN",$(SYSTEM),$$(cat $(LOGTEMPFILE)),$(LOG_NOTE),$(notdir $(@:.FOR.run=.log)))
	-@$(if $(LOG), rm $(LOGTEMPFILE))


##################################################
# Running only no compile rules
##################################################
# run c app rule
%.c.runonly:
	$(call log_section_header,"RUN",$(SYSTEM),$(@:.runonly=),$(LOG_NOTE),$(OMP_VERSION),$(notdir $(@:.runonly=.log)))
	@echo -e $(TXTGRN)"\n\n" running previously compiled: $@ $(TXTNOC) $(if $(LOG), ${RECORD}$(notdir $(@:.runonly=.log)))
# If .../test_<envname>_env_<value>...
	$(if $(findstring _env_,$@), \
	  $(call loadModules,$(C_COMPILER_MODULE)) \
		 $(BSRUN)$(RUN_TEST) --env \
		   $(shell echo "$@" | sed -e 's@.*/@@' -e 's@test_\(.*\)_env_.*@\1@' | tr 'a-z' 'A-Z') \
		   $(shell echo "$@" | sed -e 's@.*/@@' -e 's@.*_env_\([^.]*\).*@\1@') \
		   $(@:.runonly=.o) $(VERBOSE) $(if $(LOG),$(RECORD)$(notdir $(@:.runonly=.log))\
		 && echo "PASS" > $(LOGTEMPFILE) \
		 || echo "FAIL" > $(LOGTEMPFILE)), \
	  $(call loadModules,$(C_COMPILER_MODULE)) $(BSRUN)$(RUN_TEST) $(@:.runonly=.o) $(VERBOSE) $(if $(LOG),$(RECORD)$(notdir $(@:.runonly=.log))\
		 && echo "PASS" > $(LOGTEMPFILE) \
		 || echo "FAIL" > $(LOGTEMPFILE)) \
	)
	-$(call log_section_footer,"RUN",$(SYSTEM),$$(cat $(LOGTEMPFILE)),$(LOG_NOTE),$(notdir $(@:.runonly=.log)))
	-@$(if $(LOG), rm $(LOGTEMPFILE))

# run c++ app rule
%.cpp.runonly:
	$(call log_section_header,"RUN",$(SYSTEM),$(@:.runonly=),$(LOG_NOTE),$(OMP_VERSION),$(notdir $(@:.runonly=.log)))
	@echo -e $(TXTGRN)"\n\n" running previously compiled: $@ $(TXTNOC) $(if $(LOG), ${RECORD}$(notdir $(@:.runonly=.log)))
	-$(call loadModules,$(CXX_COMPILER_MODULE)) $(BSRUN)$(RUN_TEST) $(@:.runonly=.o) $(VERBOSE) $(if $(LOG),$(RECORD)$(notdir $(@:.runonly=.log))\
		&& echo "PASS" > $(LOGTEMPFILE) \
		|| echo "FAIL" > $(LOGTEMPFILE))
	-$(call log_section_footer,"RUN",$(SYSTEM),$$(cat $(LOGTEMPFILE)),$(LOG_NOTE),$(notdir $(@:.runonly=.log)))
	-@$(if $(LOG), rm $(LOGTEMPFILE))

# run Fortran app rule
%.FOR.runonly:
	$(call log_section_header,"RUN",$(SYSTEM),$(@:.FOR.runonly=),$(LOG_NOTE),$(OMP_VERSION),$(notdir $(@:.FOR.runonly=.log)))
	@echo -e $(TXTGRN)"\n\n" running previously compiled: $@ $(TXTNOC) $(if $(LOG), ${RECORD}$(notdir $(@:.FOR.runonly=.log)))
	-$(call loadModules,$(F_COMPILER_MODULE)) $(BSRUN)$(RUN_TEST) $(@:.FOR.runonly=.o) $(VERBOSE) $(if $(LOG),$(RECORD)$(notdir $(@:.FOR.runonly=.log))\
		&& echo "PASS" > $(LOGTEMPFILE) \
		|| echo "FAIL" > $(LOGTEMPFILE))
	-$(call log_section_footer,"RUN",$(SYSTEM),$$(cat $(LOGTEMPFILE)),$(LOG_NOTE),$(notdir $(@:.FOR.runonly=.log)))
	-@$(if $(LOG), rm $(LOGTEMPFILE))

# Creates the BINDIR folder
$(BINDIR):
	mkdir $@

$(LOGDIR):
	mkdir $@

$(RESULTS_CSV_OUTPUT_FILE):
	@echo "Creating $(RESULTS_CSV_OUTPUT_FILE) file"
	@echo "Currently we only support run logs that contain compilation and run outputs. Use the 'make all' rule to obtain these"
	@$(RESULTS_ANALYZER) -r -f csv -o $(RESULTS_CSV_OUTPUT_FILE) $(LOGDIRNAME)/*

$(RESULTS_JSON_OUTPUT_FILE):
	@echo "Creating $(RESULTS_JSON_OUTPUT_FILE) file"
	@echo "Currently we only support run logs that contain compilation and run outputs. Use the 'make all' rule to obtain these"
	@$(RESULTS_ANALYZER) -r -f json -o $(RESULTS_JSON_OUTPUT_FILE) $(LOGDIRNAME)/*

.PHONY: report_csv
report_csv: $(RESULTS_CSV_OUTPUT_FILE)
	@echo " === REPORT DONE === "

.PHONY: report_json
report_json: $(RESULTS_JSON_OUTPUT_FILE)
	@echo " === REPORT DONE === "

.PHONY: report_summary
report_summary:
	@$(RESULTS_ANALYZER) -r -f summary $(LOGDIRNAME)/*

.PHONY: report_html
report_html: $(RESULTS_JSON_OUTPUT_FILE) $(RESULTS_CSV_OUTPUT_FILE)
	@if [ -d "./$(RESULTS_HTML_OUTPUT_FOLDER)" ]; then \
    echo "A report exist already. Please move it before creating a new one"; \
	 else \
	  echo " === CREATING REPORT === "; \
		mkdir $(RESULTS_HTML_OUTPUT_FOLDER); \
		echo " folder $(RESULTS_HTML_OUTPUT_FOLDER) created"; \
	  cp -r $(RESULTS_HTML_REPORT_TEMPLATE)/* $(RESULTS_HTML_OUTPUT_FOLDER); \
		echo " template copied"; \
		mv $(RESULTS_CSV_OUTPUT_FILE) $(RESULTS_HTML_OUTPUT_FOLDER); \
		mv $(RESULTS_JSON_OUTPUT_FILE) $(RESULTS_HTML_OUTPUT_FOLDER); \
		sed -i "1s/.*/var jsonResults = \[/g" $(RESULTS_HTML_OUTPUT_FOLDER)/$(RESULTS_JSON_OUTPUT_FILE); \
		sed -i "$$ s/.*/];/g" $(RESULTS_HTML_OUTPUT_FOLDER)/$(RESULTS_JSON_OUTPUT_FILE); \
		echo " json file processed"; \
	fi;

	@echo " === REPORT DONE === "

ifdef REPORT_ONLINE_TAG
  # Check if the length is appropriate
  ifneq ("$(shell echo ${REPORT_ONLINE_TAG} | wc -m | grep -oh '[0-9]\+')", "10")
    $(error "REPORT_ONLINE_TAG is a 9 digit hex value. Not 9 digits")
  endif
  # Check if it's a hex value
  ifneq ("$(shell echo ${REPORT_ONLINE_TAG} | grep '[^a-f0-9]')" , "")
    $(error "REPORT_ONLINE_TAG is a 9 digit hex value. Not a Hex")
  endif
endif

ifdef REPORT_ONLINE_APPEND
  # Check if the length is appropriate
  ifndef REPORT_ONLINE_TAG
    $(error "In order to append to an online report, it is necessary to have an REPORT_ONLINE_TAG")
  endif
endif
.PHONY: report_online
report_online: $(RESULTS_JSON_OUTPUT_FILE)
	@echo " === SUBMITTING ONLINE REPORT === ";
	@FLAGS=""; \
		if [ "${REPORT_ONLINE_TAG}" != "" ]; then \
			FLAGS="$$FLAGS -t ${REPORT_ONLINE_TAG}"; \
		fi; \
		if [ "1" == "${REPORT_ONLINE_APPEND}" ]; then \
			FLAGS="$$FLAGS -a"; \
		fi; \
		date >> recent_REPORT_ONLINE_tags; \
		${REPORT_ONLINE_CONNECTION} $$FLAGS ${RESULTS_JSON_OUTPUT_FILE} | tee -a recent_REPORT_ONLINE_tags;
	@echo " This tool is for visualization purposes. "
	@echo " Our data retention policy is of 1 month. "
	@echo " After this time, we do not guarantee this link will work anymore"
	@echo " === SUBMISSION DONE === "

.PHONY: clean
clean: clear_fortran_mod
	- rm -rf $(BINDIR)

.PHONY: clear_fortran_mod
clear_fortran_mod:
	- rm -f ./ompvv/*.mod

.PHONY: tidy
tidy: clean
	- rm -rf $(LOGDIRNAME)
	- rm -rf $(RESULTS_HTML_OUTPUT_FOLDER)
	- rm -rf $(RESULTS_JSON_OUTPUT_FILE)
	- rm -rf $(RESULTS_CSV_OUTPUT_FILE)
	- rm -rf ./*.exitstatus*
	- rm -rf *.mod
	- rm -rf slurm-*

.PHONY: compilers
compilers:
	@echo "C compilers: "$(CCOMPILERS)
	@echo "C++ compilers: "$(CXXCOMPILERS)
	@echo "FORTRAN compilers: "$(FCOMPILERS)

.PHONY: help
help:
	@echo "OpenMP Offloading Validation Suite"
	@echo ""
	@echo " === USE ==="
	@echo "  make CC=ccompiler CXX=cppcompiler FC=fortrancompiler [OPTIONS] [RULE]"
	@echo ""
	@echo " === OPTIONS === "
	@echo "  VERBOSE=1                 Enables output of the commands that the make process executes"
	@echo "  VERBOSE_TESTS=1           Enables extra information display in the tests"
	@echo "  LOG=1                     Enables dump of the make process output into logs.txt"
	@echo "  LOG_ALL=1                 Enables dump of the make process output, errors, and binary execution outputs into logs.txt"
	@echo "  SYSTEM=sys_name           Includes the definitions for the requires modules and batch schedulers in the different systems"
	@echo "                            This definitions must be in sys/SYSTEM.def. (Do not include the .def extension)"
	@echo "  DEVICE_TYPE=dev_name      Specifies device type being used, either nvidia or amd, to change compiler flags as needed."
	@echo "  MODULE_LOAD=1             Before compiling or running, module load is called"
	@echo "  ADD_BATCH_SCHED=1         Add the jsrun command before the execution of the running script to send it to a compute node"
	@echo "  NO_OFFLOADING=1           Turn off offloading"
	@echo "  LINK_OMPVV_LIB=1          Link the OMPVV static library (in ompvv folder) and run the static lib test"
	@echo "  NUM_THREADS_HOST=n        Specify n, requested number of threads for tests that use num_threads() on host constructs"
	@echo "  NUM_THREADS_DEVICE=n      Specify n, requested number of threads for tests that use num_threads() on device constructs"
	@echo "  NUM_TEAMS_DEVICE=n        Specify n, requested number of threads for tests that use num_teams() on device constructs"
	@echo "  SOURCES=file or exp       Specify the source file(s) that you want to apply the rule to. You can use wildchars to select a subset of tests"
	@echo "  OMP_VERSION=[e.g. 4.5]    This specifies which version of the specs we are targeting. This version should have its folder in tests/[version]"
	@echo "                            default value is 4.5"
	@echo "                            WARNING: WHEN CHANGING VERSIONS START FROM A CLEAN BUILD. OTHERWISE BINARIES and LOG folders may collide."
	@echo "  REPORT_ONLINE_TAG=1       Allows to control the 9 digit hex value that refers to an already existing online report"
	@echo "  REPORT_ONLINE_APPEND=1    Allows to append to an already existing report. It requires a tag for a reeport (9 digit hex value)"
	@echo ""
	@echo " === RULES ==="
	@echo "  all"
	@echo "    Build and run SOURCES. If none is specified build and run all the OpenMP test files"
	@echo "  run"
	@echo "    run tests previously build. Filter tests SOURCES list, or all the OpenMP tests that are available within bin/ directory"
	@echo "  compile"
	@echo "    Compile the specific SOURCES files. If none is specified compile all the OpenMP test files"
	@echo "  clean"
	@echo "    Remove all executables from bin/ directory"
	@echo "  tidy"
	@echo "    Remove all log files, reports, and executable code (implies clean)"
	@echo "  compilers"
	@echo "    Shows available compiler configuration"
	@echo "  report_csv"
	@echo "    create a csv file containing the results existing in the logs files inside the $(LOGDIRNAME) folder"
	@echo "    currently we only support runs that contain output for compile and run"
	@echo "  report_json"
	@echo "    create a json file containing the results existing in the logs files inside the $(LOGDIRNAME) folder"
	@echo "    currently we only support runs that contain output for compile and run"
	@echo "  report_summary"
	@echo "    Create a summarized report of the tests that failed, and how many runs in total the given log folder '$(LOGDIRNAME)' has."
	@echo "  repoort_online"
	@echo "    Upload the resulting JSON file to the ompvvsollve website for easy visualization and sharing."
	@echo "  report_html"
	@echo "    create an html based results report. This rule takes the json file and a prebuild template and creates"
	@echo "    the $(RESULTS_HTML_OUTPUT_FOLDER) folder containing the report. This report allows filtering the results"
	@echo "    by system, compiler, and pass/fail result. It also allows to see the output of each tests"
	@echo ""
	@echo " === EXAMPLES ==="
	@echo "  make CC=gcc CXX=g++ FC=gfortran all         ==> compile and run all test cases with GCC"
	@echo "  make CC=gcc SOURCES=a.c all                 ==> compile and run a.c with gcc"
	@echo "  make CXX=g++ SOURCES=a.cpp all              ==> compile and run a.cpp with g++"
	@echo "  make CXX=g++ SOURCES=tests/target/* all     ==> compile and run all tests/target tests"
	@echo "  make CXX=g++ SOURCES=tests/target/*.F90 all ==> compile and run all fortran tests in tests/target"
	@echo "  make FC=gfortran SOURCES=a.F90 all          ==> compile and run a.cpp with g++"
	@echo "  make CC=xlc CXX=xlc++ FC=gfortran compile   ==> compile all test cases with XL"
	@echo "  make run                                    ==> run all the cases that exist inside bin/"
	@echo "  make report_html                            ==> Using the logs file created with the LOG option, create a report"
	@echo "  make SOURCES=myTestSource run               ==> run myTestSource if it was previously compiled "
	@echo ""
