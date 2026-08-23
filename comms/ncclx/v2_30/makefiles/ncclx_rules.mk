# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# NCCLX-only object build rules for the OSS `make` build: the .cpp / .c /
# .cu pattern rules for NCCLX sources pulled in via ncclx_sources.mk (ynl
# .cpp/.c, ctran/prims .cu, etc.). Split out of src/Makefile so the forked
# upstream makefile carries only a one-line `include` hook and stays close
# to pristine NCCL. The upstream .cc rule stays in src/Makefile.
#
# Included by src/Makefile after OBJDIR is defined, so the $(OBJDIR)/%.o
# and $(OBJDIR)/%.cu.o pattern targets expand exactly as when these rules
# were inline.

$(OBJDIR)/%.o : %.cpp $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

$(OBJDIR)/%.o : %.c $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CC) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -c $< -o $@
	@$(CC) -I. -I$(INCDIR) -I$(OBJDIR)/include $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

$(OBJDIR)/%.cu.o : %.cu $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCUFLAGS) -Xcompiler -fPIC -Xcompiler -fvisibility=hidden -I. -I$(INCDIR) -I$(OBJDIR)/include $(INCLUDES) -I$(INCPLUGIN) -I$(DOCA_HOME)/include -c $< -o $@

# NCCLX public-header install rule (ncclx.h is NCCLX-only; upstream has no
# such rule). A specific target, so it wins over the $(INCDIR)/%.h pattern
# rule regardless of definition order.
$(INCDIR)/ncclx.h : include/ncclx.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@
