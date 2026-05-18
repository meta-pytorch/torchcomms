import abc
import sys
import json
import traceback
import os
import subprocess

### SOLLVE_API_RELATED
class SOLLVE_API_01(metaclass=abc.ABCMeta):
  URL = None
  VER = None
  CREATE_TAG_URL = None
  OBTAIN_RESULT_URL = None
  UPDATE_RESULT_URL = None
  APPEND_RESULT_URL = None
  DELETE_RESULT_URL = None
  RESULT_REPORT_URL = None
  DEBUG = -1

  def __init__(self):
    self.print_log("Creating SOLLVE_API_O1", 4)
    self.URL="https://crpl.cis.udel.edu/ompvvsollve/result_report/api/"
    self.VER="0.1"
    self.CREATE_TAG_URL=self.URL+self.VER+'/create_tag'
    self.OBTAIN_RESULT_URL=self.URL+self.VER+'/obtain_result/'
    self.UPDATE_RESULT_URL=self.URL+self.VER+'/update_result/'
    self.APPEND_RESULT_URL=self.URL+self.VER+'/append_result/'
    self.DELETE_RESULT_URL=self.URL+self.VER+'/delete_result/'
    self.RESULT_REPORT_URL="https://crpl.cis.udel.edu/ompvvsollve/result_report/results.html"

  @abc.abstractmethod
  def create_tag(self):
    pass

  @abc.abstractmethod
  def obtain_result(self, tag):
    pass

  @abc.abstractmethod
  def update_result(self, tag, jsonLog):
    pass

  @abc.abstractmethod
  def append_result(self, tag, jsonLog):
    pass

  @abc.abstractmethod
  def delete_result(self,tag):
    pass

  def enable_debug(self, level=1):
    self.DEBUG = level

  def disable_debug(self):
    self.DEBUG = -1

  def print_log(self, msg:str, level=1):
    if (self.DEBUG > level):
      print("=[DEBUG]= "+msg)


## This class if requests is not installed in the system
class SOLLVE_API_01_curl(SOLLVE_API_01):
  def __init__(self):
    super().__init__()
    self.print_log("Using the CURL Back end", 1)
    #check if curl exists
    stream = os.popen('which curl')
    if (len(stream.read()) != 0):
      print("We are using CURL because we could not find the `requests` package")
      print("Error handling is limted. Please consider installing `requests` through")
      print("    pip install requests")
    else:
      raise Exception("""
            We could not find curl or the 'requests' python module in your system.
            To obtain requests use pip install requests
            """)
  def create_tag(self):
    self.print_log("create_tag curl")
    stream = subprocess.Popen(['curl', '-s', '--show-error', self.CREATE_TAG_URL], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    errors = stream.stderr.read()
    if len(errors) != 0:
      print(f"Error creating tag {errors}", file=sys.stderr)
    else:
      return str(stream.stdout.read().decode('utf-8'))

  def obtain_result(self, tag):
    self.print_log("obtain_result curl")
    stream = subprocess.Popen(['curl', '-s', '--show-error', self.OBTAIN_RESULT_URL+tag], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    errors = stream.stderr.read()
    if len(errors) != 0:
      print(f"Error creating tag {errors}", file=sys.stderr)
    else:
      # This method return scaped double quotes and other characters
      val = stream.stdout.read().decode('unicode_escape')
      # We need to remove any leading or ending spaces
      val = val.lstrip().rstrip()
      # Sometimes text comes with double quotes. Remove them
      val = val[1:] if val[0] == '"' else val
      val = val[:-1] if val[-1] == '"' else val
      return val

  def update_result(self, tag, jsonLog):
    self.print_log("update_result curl")
    command = ['curl', '-s', '--show-error', '-X POST', '-H "Content-Type: application/json"', f'-d @{jsonLog}', self.UPDATE_RESULT_URL+tag]
    command = " ".join(command)
    stream = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE,)
    errors = stream.stderr.read()
    if len(errors) != 0:
      print(f"Error creating tag {errors}", file=sys.stderr)
    else:
      # This method return scaped double quotes and other characters
      val = stream.stdout.read().decode('unicode_escape')
      # We need to remove any leading or ending spaces
      val = val.lstrip().rstrip()
      if "true" in val:
        return True
      else:
        return False

  def append_result(self, tag, jsonLog):
    self.print_log("append_result curl")
    command = ['curl', '-s', '--show-error', '-X POST', '-H "Content-Type: application/json"', f'-d @{jsonLog}', self.APPEND_RESULT_URL+tag]
    command = " ".join(command)
    stream = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE,)
    errors = stream.stderr.read()
    if len(errors) != 0:
      print(f"Error creating tag {errors}", file=sys.stderr)
    else:
      # This method return scaped double quotes and other characters
      val = stream.stdout.read().decode('unicode_escape')
      # We need to remove any leading or ending spaces
      val = val.lstrip().rstrip()
      if "true" in val:
        return True
      else:
        return False

  def delete_result(self,tag):
    self.print_log("delete_result curl")
    command = ['curl', '-s', '--show-error', '-X DELETE', self.DELETE_RESULT_URL+tag]
    command = " ".join(command)
    stream = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE,)
    errors = stream.stderr.read()
    if len(errors) != 0:
      print(f"Error creating tag {errors}", file=sys.stderr)
    else:
      # This method return scaped double quotes and other characters
      val = stream.stdout.read().decode('unicode_escape')
      # We need to remove any leading or ending spaces
      val = val.lstrip().rstrip()
      if "true" in val:
        return True
      else:
        return False


## This class if request is installed in the system
class SOLLVE_API_01_requests(SOLLVE_API_01):
  def __init__(self):
    super().__init__()
    self.print_log("Using the requests module back end", 1)
    
  def create_tag(self):
    self.print_log("create_tag requests")
    try:
      response = requests.get(self.CREATE_TAG_URL)
      # Connection went well
      if (response.status_code == 200):
        return response.text
      else:
        raise requests.exceptions.HTTPError(f"Status code not as expected {response.status_code}")
    except Exception as err: 
      print(f"Error obtaining tag [{type(err).__name__}]: {err}",file=sys.stderr)


  def obtain_result(self, tag):
    self.print_log("obtain_result requests")
    try:
      response = requests.get(self.OBTAIN_RESULT_URL+tag)
      if (response.status_code == 200):
        return response.json()
      else:
        raise requests.exceptions.HTTPError(f"Status code not as expected {response.status_code}")
    except Exception as err: 
      print(f"Error obtaining tag [{type(err).__name__}]: {err}",file=sys.stderr)

  def update_result(self, tag, jsonLog):
    self.print_log("update_result requests")
    try:
      with open(jsonLog) as f:
        try:
          json_data = json.load(f)
        except Exception as err:
          print(f"Error reading json file {jsonLog}. [{type(err).__name__}] {err}")
        response = requests.post(self.UPDATE_RESULT_URL+tag, headers={'Content-Type': 'application/json'}, json=json_data)
        if (response.status_code == 200):
          if ("true" in response.text):
            return True
          else:
            return False
        else:
          raise requests.exceptions.HTTPError(f"Status code not as expected {response.status_code}")
    except Exception as err: 
      print(f"Error obtaining tag [{type(err).__name__}]: {err}",file=sys.stderr)
      print(traceback.format_exc())
    return False

  def append_result(self, tag, jsonLog):
    self.print_log("append_result requests")
    try:
      with open(jsonLog) as f:
        try:
          json_data = json.load(f)
        except Exception as err:
          print(f"Error reading json file {jsonLog}. [{type(err).__name__}] {err}")
        response = requests.post(self.APPEND_RESULT_URL+tag, headers={'Content-Type': 'application/json'}, json=json_data)
        if (response.status_code == 200):
          if ("true" in response.text):
            return True
          else:
            return False
        else:
          raise requests.exceptions.HTTPError(f"Status code not as expected {response.status_code}")
    except Exception as err: 
      print(f"Error obtaining tag [{type(err).__name__}]: {err}",file=sys.stderr)
      print(traceback.format_exc())
    return False

  def delete_result(self,tag):
    self.print_log("delete_result requests")
    try:
      response = requests.delete(self.DELETE_RESULT_URL+tag)
      if (response.status_code == 200):
        if ("true" in response.text):
          return True
        else:
          return False
      else:
        raise requests.exceptions.HTTPError(f"Status code not as expected {response.status_code}")
    except Exception as err: 
      print(f"Error obtaining tag [{type(err).__name__}]: {err}",file=sys.stderr)

try:
  import requests
  api = SOLLVE_API_01_requests()
except:
  api = SOLLVE_API_01_curl()