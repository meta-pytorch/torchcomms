#!/usr/bin/env python
#
# Copyright (c) 2008 Google, Inc.
# Contributed by Arun Sharma <arun.sharma@google.com>
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# 

import os
from perfmon import *

def public_members(self):
    s = "{ "
    for k, v in self.__dict__.iteritems():
      if not k[0] == '_':
        s += "%s : %s, " % (k, v)
    s += " }"
    return s

class System:
  def __init__(self):
    self.ncpus = os.sysconf('SC_NPROCESSORS_ONLN')
    self.pmu = PMU()

  def __repr__(self):
    return public_members(self)

class Event:
  def __init__(self):
    pass

  def __repr__(self):
    return '\n' + public_members(self)

class EventMask:
  def __init__(self):
    pass

  def __repr__(self):
    return '\n\t' + public_members(self)

class PMU:
  def __init__(self):
    pfm_initialize()

    self.type = pfm_py_get_pmu_type()
    self.name = pfm_get_pmu_name(PFMON_MAX_EVTNAME_LEN)[1]
    self.width = pfm_py_get_hw_counter_width()

    # What does the PMU support?
    self.__implemented_pmcs = pfmlib_regmask_t()
    self.__implemented_pmds = pfmlib_regmask_t()
    self.__implemented_counters = pfmlib_regmask_t()
    pfm_get_impl_pmcs(self.__implemented_pmcs)
    pfm_get_impl_pmds(self.__implemented_pmds)
    pfm_get_impl_counters(self.__implemented_counters)
    self.implemented_pmcs = self.__implemented_pmcs.weight()
    self.implemented_pmds = self.__implemented_pmds.weight()
    self.implemented_counters = self.__implemented_counters.weight()

    self.__events = None

  def __parse_events(self):
    nevents = pfm_py_get_num_events()
    self.__events = []
    for idx in range(0, nevents):
      e = Event()
      e.name = pfm_py_get_event_name(idx)
      e.code = pfm_py_get_event_code(idx)
      e.__counters = pfmlib_regmask_t()
      pfm_get_event_counters(idx, e.__counters)
      # Now the event masks
      e.masks = []
      nmasks = pfm_py_get_num_event_masks(idx)
      for mask_idx in range(0, nmasks):
        em = EventMask()
        em.name = pfm_py_get_event_mask_name(idx, mask_idx)
        em.code = pfm_py_get_event_mask_code(idx, mask_idx)
        em.desc = pfm_py_get_event_mask_description(idx, mask_idx)
        e.masks.append(em)
      self.__events.append(e)

  def events(self):
    if not self.__events:
      self.__parse_events()
    return self.__events

  def __repr__(self):
    return public_members(self)

if __name__ == '__main__':
  from perfmon import *
  s = System()
  print s
  print s.pmu.events()
