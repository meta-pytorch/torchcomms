#!/usr/bin/env python3

import argparse, glob, sys
import os, json
import traceback


class testResult:
  ''' 
  Class for storing test results after they are obtained from the log file
  ''' 

  # Attributes

  # Test parameters
  testName = ""
  testPath = ""
  compilerName = ""
  compilerCommand = ""
  testSystem = ""
  testComments = ""
  testGitCommit = ""
  ompVersion = ""

  # Compiler results
  startingCompilerDate = ""
  endingCompilerDate = ""
  compilerPass = ""
  compilerOutput = ""

  # Runtime results
  runtimeOnly = False
  binaryPath = ""
  startingRuntimeDate = ""
  endingRuntimeDate = ""
  runtimePass = ""
  runtimeOutput = ""
  
  def __init__(self):
    # Test parameters
    testName = ""
    testPath = ""
    compilerName = ""
    compilerCommand = ""
    testGitCommit = ""
    ompVersion = ""

    # Compiler results
    startingCompilerDate = ""
    endingCompilerDate = ""
    compilerPass = ""
    compilerOutput = ""

    # Runtime results
    runtimeOnly = False
    binaryPath = ""
    startingRuntimeDate = ""
    endingRuntimeDate = ""
    runtimePass = ""
    runtimeOutput = ""

  def setTestParameters(self, newTestName, newTestPath=None, newCompilerName=None, newCompilerCommand=None, newTestCommit=None, newOMPVersion=None):
    self.testName = newTestName
    runtimeOnly = (not newTestPath)
    if newTestPath:  self.testPath = newTestPath 
    if newCompilerName:  self.compilerName = newCompilerName 
    if newCompilerCommand:  self.compilerCommand = newCompilerCommand 
    if newTestCommit:  self.testGitCommit = newTestCommit 
    if newOMPVersion:  self.ompVersion = newOMPVersion

  def setCompilerInit(self, newStartingCompilerDate, newSystem):
    self.startingCompilerDate = newStartingCompilerDate
    if newSystem:  self.testSystem = newSystem 

  def setRuntimeInit(self, newBinaryPath, newStartingRuntimeDate, newSystem):
    self.binaryPath = newBinaryPath
    self.startingRuntimeDate= newStartingRuntimeDate
    if newSystem:  self.testSystem = newSystem 

  def setCompilerResult(self, itPassed, outputText, newEndingCompilerDate, newComments=None):
    ''' setters for compiler results'''
    self.compilerPass = itPassed
    self.compilerOutput = outputText
    self.endingCompilerDate = newEndingCompilerDate
    if newComments:  self.testComments = newComments 

  def setRuntimeResult(self, itPassed, outputText, newEndingRuntimeDate, newComments=None):
    ''' setters for runtime results'''
    self.runtimePass = itPassed
    self.runtimeOutput = outputText
    self.endingRuntimeDate = newEndingRuntimeDate
    if newComments:  self.testComments = newComments 
  
  def makePathRelative(self, basePath=None):
    if (os.path.isabs(self.testPath)):
      if (basePath):
        self.testPath = os.path.relpath(self.testPath, basePath)
      else:
          self.testPath = os.path.relpath(self.testPath)

  def convert2CSV(self):
    ''' Comma Separated Values printing '''
    return '"%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s"' \
           % (self.testSystem.replace('\n',''),self.testName.replace('\n',''), self.testPath.replace('\n',''), self.ompVersion.replace('\n',''),
           self.compilerName.replace('\n',''), self.compilerCommand.replace('\n',''), self.startingCompilerDate.replace('\n',''),
           self.endingCompilerDate.replace('\n',''), self.compilerPass.replace('\n',''), self.compilerOutput.replace('\n',''),
           str(self.runtimeOnly), self.binaryPath.replace('\n',''), self.startingRuntimeDate.replace('\n',''),
           self.endingRuntimeDate.replace('\n',''), self.runtimePass.replace('\n',''), self.runtimeOutput.replace('\n',''), 
           self.testGitCommit.replace('\n',''), self.testComments.replace('\n', ''))

  def convert2dict(self):
    ''' convert to dictionary, easier to jsonify '''
    return {
        "Test name": self.testName, 
        "Test path": self.testPath,
        "Test system": self.testSystem,
        "Test comments": self.testComments,
        "Test gitCommit": self.testGitCommit,
        "OMP version": self.ompVersion,
        "Runtime only": self.runtimeOnly,
        "Compiler name": self.compilerName,
        "Compiler result": self.compilerPass,
        "Compiler output": self.compilerOutput,
        "Compiler command": self.compilerCommand,
        "Compiler starting date": self.startingCompilerDate,
        "Compiler ending date": self.endingRuntimeDate,
        "Binary path": self.binaryPath,
        "Runtime result": self.runtimePass,
        "Runtime output": self.runtimeOutput,
        "Runtime starting date": self.startingRuntimeDate,
        "Runtime ending date": self.endingRuntimeDate
    }
  def __str__(self):
    return """ 
      # Test values
      testName = "%s"
      testPath = "%s"
      testSystem = "%s"
      testComments = "%s"
      testGitCommit = "%s"
      ompVersion = "%s"
      compilerName = "%s"
      compilerCommand = "%s"

      # Compiler results
      startingCompilerDate = "%s"
      endingCompilerDate = "%s"
      compilerPass = "%s"
      compilerOutput = "%s"

      # Runtime results
      runtimeOnly = %s
      binaryPath = "%s"
      startingRuntimeDate = "%s"
      endingRuntimeDate = "%s"
      runtimePass = "%s"
      runtimeOutput = "%s"
    """ % (self.testName, self.testPath, self.testSystem, self.testComments, 
           self.testGitCommit, self.ompVersion, self.compilerName,
           self.compilerCommand, self.startingCompilerDate, 
           self.endingCompilerDate, self.compilerPass, self.compilerOutput,
           str(self.runtimeOnly), self.binaryPath, self.startingRuntimeDate, 
           self.endingRuntimeDate, self.runtimePass, self.runtimeOutput)
  def __repr__(self):
    return str(self) # End of class definition

def parseFile(log_file):
  ''' Function to parsing a single file. given the filename
  it will open the file and obtain all the information, creating an
  array of results
  '''
  returned_value = [];
  ignore_run_information = [];
  #check if log_file is string
  if isinstance(log_file, str):
    #check if log_file is a file that exist
    if os.path.isfile(log_file):
      current_state = "END"
      current_buffer = ""
      current_test = testResult()
      for line in open(log_file,'r', encoding='utf-8', errors='ignore'):
        if line.startswith("*-*-*"):
          # header line
          header_info = interpretHeader(line)
          if header_info["type"] == "COMPILE":
            # We are starting a compiler section
            current_test.setTestParameters(header_info["testName"], header_info["file"], header_info["compiler"], header_info["compilerCommand"], header_info["gitCommit"], header_info["ompVersion"])
            current_test.setCompilerInit(header_info["date"], header_info["system"])
            current_state = header_info["type"]
          elif header_info["type"] == "RUN":
            # we are starting a runtime section
            if (current_test.testName == ""):
              # This is needed to ignore tests that failed at compilation time. but sometimes we only have 
              # RUN information, and in this case we want to add the info to the logs
              if (header_info["testName"] in ignore_run_information):
                  continue

              # Get commit if it is run only information
              if (current_test.testGitCommit == ""):
                current_test.testGitCommit = header_info["gitCommit"]
              current_test.setTestParameters(header_info["testName"], newOMPVersion=header_info["ompVersion"])
            current_test.setRuntimeInit(header_info["file"], header_info["date"], header_info["system"])
            current_state = header_info["type"]
          elif header_info["type"] == "END":
            # We are ending a section
            if current_state == "COMPILE":
              current_test.setCompilerResult(header_info["result"], current_buffer, header_info["date"], header_info["comments"])
              if header_info["result"].find("FAIL") != -1:
                returned_value.append(current_test)
                # add test name to ignore run information. 
                ignore_run_information.append(current_test.testName)
                # Runtime is the last thing that should happen
                current_test = testResult()

            elif current_state == "RUN":
              current_test.setRuntimeResult(header_info["result"], current_buffer, header_info["date"], header_info["comments"])
              if current_test.compilerPass.find("FAIL") == -1:
                returned_value.append(current_test)
                # Runtime is the last thing that should happen
                current_test = testResult()

            # reset the values
            current_state = header_info["type"]
            current_buffer = ""
        else:
          # line is just output
          if current_state != "END":
            current_buffer = current_buffer + line

    else: 
      raise ValueError(str(log_file) + " is not a file")
  else: 
    raise ValueError(str(log_file) + " is not a string")
      
  return returned_value
# end of parseFile function definition

def interpretHeader(header):
  ''' Function to split a header into a dictionary containing
  the type of header, and its values'''

  # This is what is returned. It has at least a type and a date
  returned_value = { "type": "", "date": ""}
  if isinstance(header, str):
    header_split = header.split("*-*-*")[1:] # first element always empty
    # get the date, system, commit id, and OpenMP spec version
    returned_value["date"] = header_split[2]
    returned_value["system"] = header_split[3]
    returned_value["gitCommit"] = header_split[6]
    returned_value["ompVersion"] = header_split[7]
    if header_split[0].startswith("BEGIN"):
      if header_split[1].startswith('COMPILE'):
        # case when the header is a compiler header.
        # Example of a compiler line is:
        #  *-*-*BEGIN*-*-*COMPILE CC=gcc -I./ompvv -O3 -std=c99 -fopenmp -foffload=-lm -lm*-*-*Tue Nov 26 17:46:58 EST 2019*-*-*summit*-*-*tests/4.5/offloading_success.c*-*-*gcc 8.1.1*-*-*463391f*-*-*4.5*-*-*

        returned_value["type"] = "COMPILE"
        returned_value["file"] = header_split[4]
        returned_value["testName"] = os.path.basename(header_split[4])

        # remove the "COMPILE " part
        compilation_info = header_split[1][8:]
        returned_value["compiler"] = header_split[5]
        if compilation_info[:2] == "CC":
          returned_value["compilerCommand"] = compilation_info[3:]
        elif compilation_info[:3] == "CPP":
          returned_value["compilerCommand"] = compilation_info[4:]
        elif compilation_info[:1] == "F":
          returned_value["compilerCommand"] = compilation_info[2:]
        else:
          returned_value["compilerCommand"] = "undefined"
      elif header_split[1].startswith('RUN'):
        # case when the header is a runtime header.
        # Example of a runtime line is:
        # *-*-*BEGIN*-*-*RUN*-*-*Tue Nov 26 17:46:59 EST 2019*-*-*summit*-*-*bin/offloading_success.c*-*-*none*-*-*463391f*-*-*4.5*-*-*
        returned_value["type"] = "RUN"
        returned_value["file"] = header_split[4]
        returned_value["testName"] = os.path.basename(header_split[4])
    elif header_split[0].startswith('END'):
      # case when the header is a closing header.
      # Example of a closing line is:
      # *-*-*END*-*-*RUN*-*-*Tue Nov 26 17:47:00 EST 2019*-*-*summit*-*-*PASS*-*-*none*-*-*463391f*-*-*
      returned_value["type"] = "END"
      returned_value["result"] = header_split[4]
      returned_value["comments"] = header_split[5]
    else:
      # For some reason, none of the above. Should not happen
      returned_value["type"] = "undefined"
  else:
    raise ValueError("non string sent to parseHeader")

  return returned_value
# end of parseHeader function definition

def main():
  ''' Arguments parsing'''
  parser = argparse.ArgumentParser(description="Process the log files from the SOLLVE OMPVV project")
  parser.add_argument('logFiles', metavar="LOGFILES", type=str, nargs='+', help='the log files to parse')
  parser.add_argument('-f', '--format', dest='format', type=str, nargs=1, default=['JSON'], help='the format to be printed [JSON or CSV]')
  parser.add_argument('-o', '--output', dest='output', type=str, nargs=1, help='output file name')
  parser.add_argument('-r', '--relative-path', dest='relativePath', action='store_true', help="Make paths relative to the path where script is executing")

  args = parser.parse_args()
  
  logFiles = args.logFiles

  results = []

  ''' Action with the arguments '''
  for logfile in logFiles:
    files = glob.iglob(logfile)
    for fileName in files: 
      results.extend(parseFile(fileName))
 
  if len(results) == 0:
    print(" ==> No log files to process")
    return

  if args.relativePath:
    for result in results:
      result.makePathRelative()

  formatedResults = []
  formatedOutput = ""

  if args.format and args.format[0].lower() == 'json':
    for result in results:
      formatedResults.append(result.convert2dict())
    formatedOutput = json.dumps(formatedResults,indent=2, sort_keys=True)
  elif args.format and args.format[0].lower() == 'csv':
    formatedOutput = "testSystem, testName, testPath, compilerName," \
           "compilerCommand, startingCompilerDate," \
           "endingCompilerDate, compilerPass, compilerOutput," \
           "runtimeOnly, binaryPath, startingRuntimeDate," \
           "endingRuntimeDate, runtimePass, runtimeOutput, gitCommit, testComments \n"
    for result in results:
      formatedOutput = formatedOutput + result.convert2CSV() + '\n'
  elif args.format and args.format[0].lower() == 'summary':
    num_tests = len(results)
    failures = []
    acceptable_extensions = (".cpp", ".c", ".f90", ".f")				   
    number_tests_by_file_type = dict.fromkeys( acceptable_extensions, 0)	   
    number_build_failures_by_file_type = dict.fromkeys( acceptable_extensions, 0) 
    number_pass_by_file_type = dict.fromkeys( acceptable_extensions, 0)
    for result in results: 
      extension=result.testName[(result.testName).rindex("."):].lower()
      number_tests_by_file_type[extension] = number_tests_by_file_type[extension] + 1
      if result.compilerPass != "PASS":
        failures.append("  " + result.testName + " on " + result.compilerName + " (compiler) ")
        number_build_failures_by_file_type[extension] = number_build_failures_by_file_type[extension] + 1
      elif result.runtimePass != "PASS":
        failures.append("  " + result.testName + " on " + result.compilerName + " (runtime) ")
      else:
        number_pass_by_file_type[extension] = number_pass_by_file_type[extension] + 1
    if len(failures) == 0:
      formatedOutput = "PASS\nChecked " + str(num_tests) + " runs"
    else:
      formatedOutput = "FAILED\nChecked " + str(num_tests) + " runs\nReported errors(" + str(len(failures)) + "):\n"
      for failure in failures:
        formatedOutput += failure+"\n"

  # Checking if output file was specified
  if args.output and args.output[0]!="":
    outputFile = open(args.output[0], 'w')
    outputFile.write(formatedOutput)
  else:
    print(formatedOutput)
    print("Condensed Summary by file type:")
    for ext in acceptable_extensions:
      if number_tests_by_file_type[ext] > 0 :
        print( "  %s pass rate: %d/%d (%d%%) [ %d build failures ]" 
               % (ext, number_pass_by_file_type[ext],
                  number_tests_by_file_type[ext],
                  number_pass_by_file_type[ext]/number_tests_by_file_type[ext]*100,
                  number_build_failures_by_file_type[ext] ) )

 
# end of main function definition
  
if __name__ == "__main__":
      main()
