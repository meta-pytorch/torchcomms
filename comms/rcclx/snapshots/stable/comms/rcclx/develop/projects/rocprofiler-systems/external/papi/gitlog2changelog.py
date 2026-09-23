#!/usr/bin/env python3
# Copyright 2008 Marcus D. Hanwell <marcus@cryos.org>
# Minor changes for NUT by Charles Lepple
# Distributed under the terms of the GNU General Public License v2 or later
#
# Updates by Treece Burgess in October of 2024:
# Add --start_commit and --fout command line arguments
# Update the re.search conditional check to actually check against a valid return value
# Correctly account for the final entry of the git log summary

import string, re, os
from textwrap import TextWrapper
import sys, argparse

def cmd_line_interface():
   """Setup for the command line interface..

   :return: The argparse.Namespace object. 
   """
   parser = argparse.ArgumentParser()
   parser.add_argument("--starting_commit", required = True, 
                       help = "Commit hash for the starting point of the desired range (non-inclusive).") 
   parser.add_argument("--fout", required = True, 
                       help = "Name to give output file. E.g. ChangeLogP800.txt")

   return parser.parse_args()

if __name__ == "__main__":
    # Collect the command line arguments
    args = cmd_line_interface()

    # Range of specific commits that we want to create a change log for
    rev_range = '%s..HEAD' % args.starting_commit

    # Execute git log with the desired command line options.
    # This is implemented using subprocess.Popen
    fin = os.popen('git log --summary --stat --no-merges --date=short %s' % rev_range, 'r', buffering = -1)
    # Needed to properly parse final entry 
    lines = fin.readlines()
    last_line = lines[-1]
    # Create a ChangeLog file in the current directory.
    fout = open(args.fout, 'w')

    # Set up the loop variables in order to locate the blocks we want
    authorFound = False
    dateFound = False
    messageFound = False
    filesFound = False
    message = ""
    messageNL = False
    files = ""
    prevAuthorLine = ""

    wrapper = TextWrapper(initial_indent="\t", subsequent_indent="\t  ") 

    # The main part of the loop
    for line in lines:
        # The commit line marks the start of a new commit object.
        if line.startswith('commit'):
            # Start all over again...
            authorFound = False
            dateFound = False
            messageFound = False
            messageNL = False
            message = ""
            filesFound = False
            files = ""
            continue
        # Match the author line and extract the part we want
        elif 'Author:' in line:
            authorList = re.split(': ', line, 1)
            author = authorList[1]
            author = author[0:len(author)-1]
            authorFound = True
        # Match the date line
        elif 'Date:' in line:
            dateList = re.split(':   ', line, 1)
            date = dateList[1]
            date = date[0:len(date)-1]
            dateFound = True
        # The Fossil-IDs are ignored:
        elif line.startswith('    Fossil-ID:') or line.startswith('    [[SVN:'):
            continue
        # The svn-id lines are ignored
        elif '    git-svn-id:' in line:
            continue
        # The sign off line is ignored too
        elif 'Signed-off-by' in line:
            continue
        # Extract the actual commit message for this commit
        elif authorFound & dateFound & messageFound == False:
            # Find the commit message if we can
            if len(line) == 1:
                if messageNL:
                    messageFound = True
                else:
                    messageNL = True
            elif len(line) == 4:
                messageFound = True
            else:
                if len(message) == 0:
                    message = message + line.strip()
                else:
                    message = message + " " + line.strip()
        # If this line is hit all of the files have been stored for this commit
        elif re.search('files? changed', line) != None:
            filesFound = True
            # We only want to continue if it is not the last line;
            # continuing on the last line would skip the final entry
            if line is not last_line:
                continue
        # Collect the files for this commit. FIXME: Still need to add +/- to files
        elif authorFound & dateFound & messageFound:
            fileList = re.split(' \| ', line, 2)
            if len(fileList) > 1:
                if len(files) > 0:
                    files = files + ", " + fileList[0].strip()
                else:
                    files = fileList[0].strip()
        # All of the parts of the commit have been found - write out the entry
        if authorFound & dateFound & messageFound & filesFound:
            # First the author line, only outputted if it is the first for that
            # author on this day
            authorLine = date + "  " + author
            if len(prevAuthorLine) == 0:
                fout.write(authorLine + "\n\n")
            elif authorLine == prevAuthorLine:
                pass
            else:
                fout.write("\n" + authorLine + "\n\n")

            # Assemble the actual commit message line(s) and limit the line length
            # to 80 characters.
            commitLine = "* " + files + ": " + message

            # Write out the commit line
            fout.write(wrapper.fill(commitLine) + "\n")

            #Now reset all the variables ready for a new commit block.
            authorFound = False
            dateFound = False
            messageFound = False
            messageNL = False
            message = ""
            filesFound = False
            files = ""
            prevAuthorLine = authorLine

    # Close the input and output lines now that we are finished.
    fin.close()
    fout.close()
