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
# Self monitoring example. Copied from self.c

import os
from optparse import OptionParser
import random
import errno
from perfmon import *

if __name__ == '__main__':
  parser = OptionParser()
  parser.add_option("-e", "--events", help="Events to use",
                    action="store", dest="events")
  (options, args) = parser.parse_args()

  s = PerThreadSession(int(os.getpid()))
  if options.events:
    events = options.events.split(",")
  else:
    raise "You need to specify events to monitor"
  s.dispatch_events(events)
  s.load()
  s.start()

  # code to be measured
  #
  # note that this is not identical to what examples/self.c does
  # thus counts will be different in the end
  for i in range(1, 10000000):
    random.random()

  s.stop()

  # read the counts
  for i in xrange(s.npmds):
    print """PMD%d\t%lu""" % (s.pmds[0][i].reg_num, s.pmds[0][i].reg_value)
