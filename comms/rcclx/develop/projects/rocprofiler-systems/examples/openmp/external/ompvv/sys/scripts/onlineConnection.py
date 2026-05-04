#!/usr/bin/env python3

import argparse, glob, sys
import json
import os
import traceback
from API.sollve_api import api

### HELPER FUNCTIIONS
DEBUG = -1
def print_log(msg:str, level):
  if (DEBUG > level):
    print("=[DEBUG]= "+msg)

if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")

def confirm(prompt=None, resp=False):
  """prompts for yes or no response from the user. Returns True for yes and
  False for no.

  'resp' should be set to the default value assumed by the caller when
  user simply types ENTER.

  >>> confirm(prompt='Create Directory?', resp=True)
  Create Directory? [y]|n: 
  True
  >>> confirm(prompt='Create Directory?', resp=False)
  Create Directory? [n]|y: 
  False
  >>> confirm(prompt='Create Directory?', resp=False)
  Create Directory? [n]|y: y
  True

  """
  
  if prompt is None:
      prompt = 'Confirm'

  if resp:
      prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
  else:
      prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')
      
  while True:
      ans = input(prompt)
      if not ans:
          return resp
      if ans not in ['y', 'Y', 'n', 'N']:
          print('please enter y or n.')
          continue
      if ans == 'y' or ans == 'Y':
          return True
      if ans == 'n' or ans == 'N':
          return False

### Program
def main():

  ''' Arguments parsing'''
  parser = argparse.ArgumentParser(description="""
  Connection manager with SOLLVE online visualization tool
      This script connects to our result visualziation tool. There are three
      use modes. The first one is to create a new visualization
  """)
  parser.add_argument('jsonLogFile', metavar="JSONFILE", type=str, nargs='*', help='the json formatted log file to be sent')
  parser.add_argument('-t', '--tag', dest='tag', type=str, nargs=1, help='Already existing tag to be appended, removed or updated')
  parser.add_argument('-d', '--debug', dest='debug', type=int, help='VERBOSE. Show debugging information for this script')
  parse_group = parser.add_mutually_exclusive_group() 
  parse_group.add_argument('-a', '--append', dest='append', action='store_true', help="Uses the tag to append to its already existing results")
  parse_group.add_argument('-r', '--remove', dest='remove', action='store_true', help="Remove the set of results completely [NON-REVERSIBLE]")
  parse_group.add_argument('-o', '--obtain', dest='obtain_file', action='store_true', help="uses the tag to obtain the JSON file from server")
  parser.add_argument('-f', '--force', dest='force_no_confirm', action='store_true', help="Allows to avoid any confirmation message (see remove)")

  # Obtaining the arguments from command line
  args = parser.parse_args()
  print_log("Using arguments = " + str(args), 3)

  # Enable debug mode and setting debug level
  if (args.debug):
    global DEBUG 
    DEBUG = args.debug
    api.enable_debug(DEBUG)

  # Checking if a tag was specified with the append, delete or obtain options
  if (args.append or args.remove or args.obtain_file) and args.tag is None:
    print("A tag is required with the --append, --delete or --obtain options", file=sys.stderr)
    return
  
  # Make sure the user wants to remove. This is irreversible
  if (args.remove and not args.force_no_confirm):
    confirmed = confirm("[WARNING] Are you sure you want to delete this tag from the servers? This is irreversible")
    if (not confirmed):
      print_log("User rejected deletion",1)
      return

  # Obtaining the tag
  tag = None
  if (args.tag):
    tag = args.tag[0]

  # Obtaining the json file to be processed
  if (args.jsonLogFile):
    jsonLogFile = args.jsonLogFile
    for logFile in jsonLogFile: 
      print_log("Using Log file " + logFile, 1)

      ## For obtain
      if (args.obtain_file):
        result = api.obtain_result(tag)
        with open(logFile, "w") as f:
          json.dump(json.loads(result), f, indent=2)

      ## For update
      if not (args.append or args.remove or args.obtain_file):
        if (not tag):
          tag = api.create_tag()

        if os.path.exists(logFile):
          if (not api.update_result(tag, logFile)):
            print("Something went wrong with the update")
        else:
          print(f"The file {logFile} does not exists")

      ## For append
      if (args.append):
        if os.path.exists(logFile):
          api.append_result(tag, logFile)
        else:
          print(f"The file {logFile} does not exists")
  elif (not args.remove):
    parser.print_usage()
    return

  ## For remove
  if (args.remove):
    result = api.delete_result(tag)

  if (not args.remove):
    print(f" Your report tag is {tag}. Do not lose this number")
    print(f" Visit your report at:\n    {api.RESULT_REPORT_URL}?result_report={tag}")

if __name__ == "__main__":
    main()