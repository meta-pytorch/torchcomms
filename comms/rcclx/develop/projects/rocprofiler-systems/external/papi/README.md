**[PAPI: The Performance Application Programming Interface](https://icl.utk.edu/exa-papi/)**

**[Innovative Computing Laboratory (ICL)](http://www.icl.utk.edu/)**

**[PAPI Wiki - Documentation](https://github.com/icl-utk-edu/papi/wiki/)**

**University of Tennessee, Knoxville (UTK)**


***
[TOC]
***

# About

The Performance Application Programming Interface (PAPI) provides tool
designers and application engineers with a consistent interface and methodology
for the use of low-level performance counter hardware found across the entire
compute system (i.e. CPUs, GPUs, on/off-chip memory, interconnects, I/O system,
energy/power, etc.). PAPI enables users to see, in near real time, the
relations between software performance and hardware events across the entire
computer system.

[The ECP Exa-PAPI project](https://icl.utk.edu/exa-papi/) builds on the latest
PAPI project and extends it with:

* Performance counter monitoring capabilities for new and advanced ECP
  hardware, and software technologies.
* Fine-grained power management support.
* Functionality for performance counter analysis at "task granularity" for
  task-based runtime systems.
* "Software-defined Events" that originate from the ECP software stack and are
  currently treated as black boxes (i.e., communication libraries, math
  libraries, task-based runtime systems, etc.)

The objective is to enable monitoring of both types of performance
events---hardware- and software-related events---in a uniform way, through one
consistent PAPI interface. Third-party tools and application developers will
have to handle only a single hook to PAPI in order to access all hardware
performance counters in a system, including the new software-defined events.


***


# Getting Help

* Visit our FAQ at: <https://icl-utk-edu.github.io/papi/PAPI_FAQ.html> 
  or read a snapshot of the FAQ in papi/PAPI_FAQ.html
* For assistance with PAPI, email ptools-perfapi@icl.utk.edu.
* You can also join the PAPI User Google group by going to
  <https://groups.google.com/a/icl.utk.edu/forum/#!forum/ptools-perfapi> 
  to read historical postings to the list.

***


# Contributing

The PAPI project welcomes contributions from new developers. Contributions can
be offered through the standard GitHub pull request model. We strongly
encourage you to coordinate large contributions with the PAPI development team
early in the process.

**For timely pull request reviews and feedback, it is important to submit 
one (1) pull request per feature / bug fix.**

In order to create a pull request on a public read-only repo, 
you will need to do the following:

1. Fork the PAPI repo (click "+" on the left and "Fork this repository").

2. Clone it.

3. Make your changes and push them.

4. Click "create pull request" from your repo (not the PAPI repo).

***


# Resources

* Visit the [Exa-PAPI website](https://icl.utk.edu/exa-papi/) to find out more
  about ongoing PAPI and
  [PAPI++](https://www.exascaleproject.org/papi-as-de-facto-standard-interface-for-performance-event-monitoring-at-the-exascale/)
  developments and research.
* Visit the [PAPI website (retired)](https://icl.utk.edu/papi/) for basic
  information about PAPI.
* Visit the [ECP website](https://www.exascaleproject.org/) to find out more
  about the DOE Exascale Computing Initiative.
* Visit the [PAPI Papers and Presentations](https://www.icl.utk.edu/view/biblio/project/papi?items_per_page=All) to find out more about PAPI papers and presentations.

***


# License

    Copyright (c) 2024, University of Tennessee
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the University of Tennessee nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
