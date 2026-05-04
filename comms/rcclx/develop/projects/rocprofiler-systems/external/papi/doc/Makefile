.PHONY:	clean clobber distclean install force_me all

all: man
	@echo "Built PAPI user documentation"

html: force_me
	doxygen Doxyfile-html

man: man/man1 man/man3

man/man3: ../src/papi.h ../src/papi.c ../src/high-level/papi_hl.c ../src/papi_fwrappers.c
	doxygen Doxyfile-man3

man/man1: ../src/utils/papi_avail.c ../src/utils/papi_clockres.c  ../src/utils/papi_command_line.c ../src/utils/papi_component_avail.c ../src/utils/papi_cost.c ../src/utils/papi_decode.c ../src/utils/papi_error_codes.c ../src/utils/papi_event_chooser.c ../src/utils/papi_xml_event_info.c ../src/utils/papi_mem_info.c ../src/utils/papi_multiplex_cost.c ../src/utils/papi_native_avail.c  ../src/utils/papi_version.c ../src/utils/papi_hardware_avail.c
	doxygen Doxyfile-man1
 
clean:
	rm -rf man html doxyerror

distclean clobber: clean

install: man
	-rm -f man/man3/HighLevelInfo.3
	-rm -f man/man3/papi_data_structures.3
	-rm -r ../man/man1/*.1 ../man/man3/*.3
	-cp -R man/man1/*.1 ../man/man1
	-cp -R man/man3/*.3 ../man/man3
