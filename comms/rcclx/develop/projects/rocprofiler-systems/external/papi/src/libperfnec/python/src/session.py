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

from perfmon import *
from linux import sched
import os
import sys
from threading import Thread
# Shouldn't be necessary for python version >= 2.5 
from Queue import Queue
 
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/425445
def once(func):
    "A decorator that runs a function only once."
    def decorated(*args, **kwargs):
        try:
            return decorated._once_result
        except AttributeError:
            decorated._once_result = func(*args, **kwargs)
            return decorated._once_result
    return decorated
 
@once
def pfm_initialize_once():
  # Initialize once
  opts = pfmlib_options_t()
  opts.pfm_verbose = 1
  pfm_set_options(opts)
  pfm_initialize()
 
# Common base class
class Session:
  def __init__(self, n):
    self.system = System()
    pfm_initialize_once()

    # Setup context
    self.ctxts = []
    self.fds = []
    self.inps = []
    self.outps = []
    self.pmcs = []
    self.pmds = []
    for i in xrange(n):
      ctx = pfarg_ctx_t()
      ctx.zero()
      ctx.ctx_flags = self.ctx_flags
      fd = pfm_create_context(ctx, None, None, 0)
      self.ctxts.append(ctx)
      self.fds.append(fd)

  def __del__(self):
    if self.__dict__.has_key("fds"):
      for fd in self.fds:
        os.close(fd)

  def dispatch_event_one(self, events, which):
    # Select and dispatch events
    inp = pfmlib_input_param_t()
    for i in xrange(0, len(events)):
      pfm_find_full_event(events[i], inp.pfp_events[i])
    inp.pfp_dfl_plm = self.default_pl
    inp.pfp_flags = self.pfp_flags
    outp = pfmlib_output_param_t()
    cnt = len(events)
    inp.pfp_event_count = cnt
    pfm_dispatch_events(inp, None, outp, None)

    # pfp_pm_count may be > cnt
    cnt = outp.pfp_pmc_count
    pmcs = pmc(outp.pfp_pmc_count)
    pmds = pmd(outp.pfp_pmd_count)
    for i in xrange(outp.pfp_pmc_count):
      npmc = pfarg_pmc_t()
      npmc.reg_num = outp.pfp_pmcs[i].reg_num
      npmc.reg_value = outp.pfp_pmcs[i].reg_value

      pmcs[i] = npmc

    self.npmds = outp.pfp_pmd_count
    for i in xrange(outp.pfp_pmd_count):
      npmd = pfarg_pmd_t()
      npmd.reg_num = outp.pfp_pmds[i].reg_num
      pmds[i] = npmd

    # Program PMCs and PMDs
    fd = self.fds[which]
    pfm_write_pmcs(fd, pmcs, outp.pfp_pmc_count)
    pfm_write_pmds(fd, pmds, outp.pfp_pmd_count)

    # Save all the state in various vectors
    self.inps.append(inp)
    self.outps.append(outp)
    self.pmcs.append(pmcs)
    self.pmds.append(pmds)

  def dispatch_events(self, events):
    for i in xrange(len(self.fds)):
      self.dispatch_event_one(events, i)

  def load_one(self, i):
    fd = self.fds[i]
    load = pfarg_load_t()
    load.zero()
    load.load_pid = self.targets[i]
    try:
      pfm_load_context(fd, load)
    except OSError, err:
      import errno
      if (err.errno == errno.EBUSY):
        err.strerror = "Another conflicting perfmon session?"
      raise err

  def load(self):
    for i in xrange(len(self.fds)):
      self.load_one(i)
    
  def start_one(self, i):
    pfm_start(self.fds[i], None)

  def start(self):
    for i in xrange(len(self.fds)):
      self.start_one(i)

  def stop_one(self, i):
    fd = self.fds[i]
    pmds = self.pmds[i]
    pfm_stop(fd)
    pfm_read_pmds(fd, pmds, self.npmds)

  def stop(self):
    for i in xrange(len(self.fds)):
      self.stop_one(i)

class PerfmonThread(Thread):
  def __init__(self, session, i, cpu):
    Thread.__init__(self)
    self.cpu = cpu
    self.session = session
    self.index = i
    self.done = 0
    self.started = 0

  def run(self):
    queue = self.session.queues[self.index]
    exceptions = self.session.exceptions[self.index]
    cpu_set = sched.cpu_set_t()
    cpu_set.set(self.cpu)
    sched.setaffinity(0, cpu_set)
    while not self.done:
      # wait for a command from the master
      method = queue.get()
      try:
        method(self.session, self.index)
      except:
        exceptions.put(sys.exc_info())
        queue.task_done()
	break
      queue.task_done()

def run_in_other_thread(func):
    "A decorator that runs a function in another thread (second argument)"
    def decorated(*args, **kwargs):
      self = args[0]
      i = args[1]
      # Tell thread i to call func()
      self.queues[i].put(func)
      self.queues[i].join()
      if not self.exceptions[i].empty():
        exc = self.exceptions[i].get()
	# Let the main thread know we had an exception
        self.exceptions[i].put(exc)
        print "CPU: %d, exception: %s" % (i, exc)
	raise exc[1]
    return decorated

class SystemWideSession(Session):
  def __init__(self, cpulist):
    self.default_pl =  PFM_PLM3 | PFM_PLM0
    self.targets = cpulist
    self.ctx_flags = PFM_FL_SYSTEM_WIDE
    self.pfp_flags = PFMLIB_PFP_SYSTEMWIDE
    self.threads = []
    self.queues = []
    self.exceptions = []
    n = len(cpulist)
    for i in xrange(n):
      t = PerfmonThread(self, i, cpulist[i])
      self.threads.append(t)
      self.queues.append(Queue(0))
      self.exceptions.append(Queue(0))
      t.start()
    Session.__init__(self, n)

  def __del__(self):
    self.cleanup()
    Session.__del__(self)

  def cleanup(self):
    for t in self.threads:
      t.done = 1
      # join only threads with no exceptions
      if self.exceptions[t.index].empty():
        if t.started:
          self.stop_one(t.index)
        else:
          self.wakeup(t.index)
        t.join()
    self.threads = []
  
  @run_in_other_thread
  def load_one(self, i):
    Session.load_one(self, i)

  @run_in_other_thread
  def start_one(self, i):
    Session.start_one(self, i)
    self.threads[i].started = 1

  @run_in_other_thread
  def stop_one(self, i):
    Session.stop_one(self, i)
    self.threads[i].started = 0

  @run_in_other_thread
  def wakeup(self, i):
    "Do nothing. Just wakeup the other thread"
    pass

class PerThreadSession(Session):
  def __init__(self, pid):
    self.targets = [pid]
    self.default_pl =  PFM_PLM3
    self.ctx_flags = 0
    self.pfp_flags = 0
    Session.__init__(self, 1)

  def __del__(self):
    Session.__del__(self)
