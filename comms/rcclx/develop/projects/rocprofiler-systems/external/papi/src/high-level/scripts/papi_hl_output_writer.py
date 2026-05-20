#!/usr/bin/env python3
## 
# @file papi_hl_output_writer.py
# @brief Converts HL output to be more comprehensible. 
# Output is enhanced by creating derived metrics like IPC,
# MFlop/s, and MFlips/s. As well as real and processor time.

from __future__ import division
from collections import OrderedDict

import argparse
import os
import json
# Make it work for Python 2+3 and with Unicode
import io
##\cond
try:
  to_unicode = unicode
except NameError:
  to_unicode = str
##\endcond

event_definitions = {}
process_num = {}

derived_metric_names = {
    'region_count':'Region count',
    'cycles':'Total elapsed cycles',
    'real_time_nsec':'Real time in s',
    'perf::TASK-CLOCK':'CPU time in s'
}

event_rate_names = OrderedDict([
    ('PAPI_FP_INS','MFLIPS/s'),
    ('PAPI_VEC_SP','Single precision vector/SIMD instructions rate in M/s'),
    ('PAPI_VEC_DP','Double precision vector/SIMD instructions rate in M/s'),
    ('PAPI_FP_OPS','MFLOPS/s'),
    ('PAPI_SP_OPS','Single precision MFLOPS/s'),
    ('PAPI_DP_OPS','Double precision MFLOPS/s')
])

def merge_json_files(source_dir):
    """!
    Function definition for merge_json_files.

    Merge multiple .json files together into a single dictionary.

    @param source_dir A directory containing one or more .json files from PAPI
                      HL function calls
  
    @returns An ordered dictionary containing measurements from recorded events 
             for one or more .json files generated from PAPI HL function calls.
    """
    json_object = {}
    events_stored = False

    #get measurement files
    file_list = os.listdir(source_dir)
    file_list.sort()
    rank_cnt = 0
    json_rank = OrderedDict()
  
    for item in file_list:
        #determine mpi rank based on file name (rank_#)
        rank = item.split('_', 1)[1]
        rank = rank.rsplit('.', 1)[0]
        try:
            rank = int(rank)
        except:
            rank = rank_cnt

        #open measurement file
        file_name = str(source_dir) + "/" + str(item)

        try:
            with open(file_name) as json_file:
                #keep order of all objects
                data = json.load(json_file, object_pairs_hook=OrderedDict)
        except IOError as ioe:
            print("Cannot open file {} ({})".format(file_name, repr(ioe)))
            return

        #store global data
        if events_stored == False:
            global event_definitions
            event_definitions = data['event_definitions']
            events_stored = True

        #get all threads
        json_rank[str(rank)] = OrderedDict()
        json_rank[str(rank)]['threads'] = data['threads']

        rank_cnt = rank_cnt + 1
  
    json_object['ranks'] = json_rank

    return json_object

def parse_source_file(source_file):
    """!
    Function definition for parse_source_file.

    Parses a single user passed .json file generated from PAPI HL function calls.

    @param source_file .json file generated from PAPI HL function calls.

    @returns An ordered dictionary containing measurements from recorded events 
             for a single .json file, generated from PAPI HL function calls.
    """
    json_data = {}
    json_rank = OrderedDict()
    events_stored = False

    # determine mpi rank based on file name (rank_#)
    rank = source_file.split('_', 1)[1]
    rank = source_file.rsplit('.', 1)[0]

    # open json file provided by user
    f = open(source_file)

    # return ordered dictionary
    data = json.load(f, object_pairs_hook = OrderedDict)

    #store global data
    if events_stored == False:
        global event_definitions
        event_definitions = data['event_definitions']
        events_stored = True

    # get all threads
    json_rank[str(rank)] = OrderedDict()
    json_rank[str(rank)]['threads'] = data['threads']

    json_data['ranks'] = json_rank 

    return json_data

class Sum_Counter(object):
    """!
    Sum_Counter class defintion.

    Calculates the min, max, median or sum for the measurements of a 
    recorded events.
    """
    def __init__(self):
        """!
        Sum_Counter class initializer.
        """
        self.min = None
        self.all_values = []
        self.max = 0

    def add_event(self, value):
        """!
        Method definition for add_event.

        Add a recorded event and measurement to summary output.

        @param value Measurement from a recorded event. E.g. PAPI_TOT_INS.
        """
        if isinstance(value, dict):
            if self.min is None or self.min > int(value['min']):
                self.min = int(value['min'])
            self.all_values.append(int(value['avg']))
            if self.max < int(value['max']):
                self.max = int(value['max'])
        else:
            val = int(value)
            if self.min is None or self.min > val:
                self.min = val
            self.all_values.append(val)
            if self.max < val:
                self.max = val

    def get_min(self):
        """!
        Method definition for get_min.
 
        Calculates the minimum for a set of measurements for a recorded event.

        @returns The minimum for a set of measurement values for a recorded event.
                 E.g. PAPI_TOT_INS.
        """
        return self.min

    def get_median(self):
        """!
        Method definition for get_median.

        Calculates the median for a set of measurements for a recorded event.

        @returns The median for a set of measurement values for a recorded event.
                 E.g. PAPI_TOT_INS. 
        """
        n = len(self.all_values)
        s = sorted(self.all_values)
        return (sum(s[n//2-1:n//2+1])/2.0, s[n//2])[n % 2] if n else None

    def get_sum(self):
        """!
        Method definition for get_sum.

        Calculates the sum for a set of measurements for a recorded event.

        @returns The sum of measurement values for a recorded event.
                 E.g. PAPI_TOT_INS.
        """
        sum = 0
        for value in self.all_values:
            sum += value
        return sum

    def get_max(self):
        """!
        Method definition for get_max.

        Calculate the maximum for a set of measurements for a recorded event.

        @returns The maximum for a set of measurement values for a recorded event.
                 E.g. PAPI_TOT_INS.
        """
        return self.max

class Sum_Counters(object):
    """!
    Sum_Counters class defintion.

    Gathers summary output for a region (e.g. computation) and accompanying 
    measurements for a recorded event (e.g. PAPI_TOT_INS).
    """

    def __init__(self):
        """!
        Sum_Counters class initializer.
        """
        self.regions = OrderedDict()
        self.regions_last_rank_id = {}
        self.regions_rank_num = {}
        self.regions_last_thread_id = {}
        self.regions_thread_num = {}

        self.clean_regions = OrderedDict()
        self.sum_counters = OrderedDict()

    def add_region(self, rank_id, thread_id, events=OrderedDict()):
        """! Method defintion for add_region.

        Adds the region (e.g. computation) and accompanying measurements for a recorded event (e.g. PAPI_TOT_INS) to summary output.

        @param rank_id MPI rank, if no MPI rank is present this value will be random.
        @param thread_id Thread identifier containing performance events. E.g. 0. 
        @param events An ordered dictionary containing measurements for recorded events obtained through PAPI HL function calls. E.g. PAPI_TOT_INS.
        """
        #remove all read values caused by PAPI_hl_read
        cleaned_events = OrderedDict()
        region_name = 'unknown'
        cleaned_events['region_count'] = 1
        for key,value in events.items():
            if 'name' in key:
                region_name = value
                continue
            if 'parent_region_id' in key:
                continue
            metric_value = value
            if isinstance(value, dict):
                if "region_value" in value:
                    metric_value = float(value['region_value'])
            cleaned_events[key] = metric_value

        #create new Sum_Counter object for each new region
        if region_name not in self.regions:
            self.regions[region_name] = {}
            self.regions_last_rank_id[region_name] = rank_id
            self.regions_last_thread_id[region_name] = thread_id
            self.regions_rank_num[region_name] = 1
            self.regions_thread_num[region_name] = 1
            self.sum_counters[region_name] = OrderedDict()
            for key,value in cleaned_events.items():
                self.sum_counters[region_name][key] = Sum_Counter()
                self.sum_counters[region_name][key].add_event(value)
        else:
            #increase number of ranks and threads when rank_id has changed
            if self.regions_last_rank_id[region_name] != rank_id:
                self.regions_last_rank_id[region_name] = rank_id
                self.regions_rank_num[region_name] += 1
                self.regions_last_thread_id[region_name] = thread_id
                self.regions_thread_num[region_name] += 1

            #increase number of threads when thread_id has changed
            if self.regions_last_thread_id[region_name] != thread_id:
                self.regions_last_thread_id[region_name] = thread_id
                self.regions_thread_num[region_name] += 1

            for key,value in cleaned_events.items():
                self.sum_counters[region_name][key].add_event(value)

        self.regions[region_name]['rank_num'] = self.regions_rank_num[region_name]
        self.regions[region_name]['thread_num'] = self.regions_thread_num[region_name]

    def get_json(self):
        """!
        Method definition for get_json.

        Calculates the min, max, median, or sum for a set of measurements for a 
        recorded event. E.g. PAPI_TOT_INS.

        @returns An ordered dictionary containing summary measurements for recorded 
                 events. E.g. PAPI_TOT_INS.
        """
        sum_json = OrderedDict()
        for name in self.regions:
            events = OrderedDict()
            for key,value in self.sum_counters.items():
                if key == name:
                    region_count = 1
                    for event_key,event_value in value.items():
                        if event_key == 'region_count':
                            events[event_key] = int(event_value.get_sum())
                            region_count = events[event_key]
                        else:
                            global event_definitions
                            if region_count > 1:
                                events[event_key] = OrderedDict()
                                if event_key == 'cycles' or event_key == 'real_time_nsec':
                                    events[event_key]['total'] = event_value.get_sum()
                                else:
                                    if ( event_definitions[event_key]['type'] == 'delta' and
                                         event_definitions[event_key]['component'] == 'perf_event' ):
                                        events[event_key]['total'] = event_value.get_sum()
                                events[event_key]['min'] = event_value.get_min()
                                events[event_key]['median'] = event_value.get_median()
                                events[event_key]['max'] = event_value.get_max()
                            else:
                                #sequential code
                                if event_key == 'cycles' or event_key == 'real_time_nsec':
                                    events[event_key] = event_value.get_min()
                                else:
                                    if event_definitions[event_key] == 'instant' and region_count > 1:
                                        events[event_key] = OrderedDict()
                                        events[event_key]['min'] = event_value.get_min()
                                        events[event_key]['median'] = event_value.get_median()
                                        events[event_key]['max'] = event_value.get_max()
                                    else:
                                        events[event_key] = event_value.get_min()
                    break

            #add number of ranks and threads in case of a parallel code
            if self.regions[name]['rank_num'] > 1 or self.regions[name]['thread_num'] > 1:
                events['Number of ranks'] = self.regions[name]['rank_num']
                events['Number of threads per rank'] = int(self.regions[name]['thread_num'] / self.regions[name]['rank_num'])
            sum_json[name] = events

            global process_num
            process_num[name] = self.regions[name]['rank_num'] * self.regions[name]['thread_num']
        return sum_json

def derive_sum_json_object(data):
    """!
    Function definition for derive_sum_json_object.

    Calculates the derived event measurements (IPC) from the recorded events 
    obtained through PAPI HL function calls.

    @param data An ordered dictionary containing the number of threads and  
                measurements for the recorded events. E.g. PAPI_TOT_INS.
  
    @returns An ordered dicitonary filled with formatted measurements for derived
             events.
    """    
    json_object = OrderedDict()

    for region_key,region_value in data.items():
        derive_events = OrderedDict()
        events = region_value.copy()

        #remember runtime for other metrics like MFLOPS
        rt = {}

        #remember region count
        region_cnt = 1

        #Region Count
        if 'region_count' in events:
            derive_events[derived_metric_names['region_count']] = int(events['region_count'])
            region_cnt = int(events['region_count'])
            del events['region_count']

        #skip cycles
        if 'cycles' in events:
            del events['cycles']

        #Real Time
        if 'real_time_nsec' in events:
            event_name = derived_metric_names['real_time_nsec']
            if region_cnt > 1:
                for metric in ['total', 'min', 'median', 'max']:
                    rt[metric] = convert_value(events['real_time_nsec'][metric], 'Runtime')
                derive_events[event_name] = rt['max']
            else:
                rt['total'] = convert_value(events['real_time_nsec'], 'Runtime')
                derive_events[event_name] = rt['total']
            del events['real_time_nsec']


        #CPU Time
        if 'perf::TASK-CLOCK' in events:
            event_name = derived_metric_names['perf::TASK-CLOCK']
            if region_cnt > 1:
                derive_events[event_name] = convert_value(events['perf::TASK-CLOCK']['total'], 'CPUtime')
            else:
                derive_events[event_name] = convert_value(events['perf::TASK-CLOCK'], 'CPUtime')
            del events['perf::TASK-CLOCK']

        #PAPI_TOT_INS and PAPI_TOT_CYC to calculate IPC
        if 'PAPI_TOT_INS' in events and 'PAPI_TOT_CYC' in events:
            event_name = 'IPC'
            metric = 'total'
            try:
                if region_cnt > 1: 
                    ipc = float(format(float(int(events['PAPI_TOT_INS'][metric]) / int(events['PAPI_TOT_CYC'][metric])), '.2f'))
                else:
                    ipc = float(format(float(int(events['PAPI_TOT_INS']) / int(events['PAPI_TOT_CYC'])), '.2f'))
            except:
                ipc = 'n/a'
            derive_events[event_name] = ipc

            del events['PAPI_TOT_INS']
            del events['PAPI_TOT_CYC']
    
        #Rates
        global event_rate_names
        for rate_event in event_rate_names:
            if rate_event in events:
                event_name = event_rate_names[rate_event]
                metric = 'total'
                try:
                    if region_cnt > 1:
                        rate = float(format(float(events[rate_event][metric]) / 1000000 / rt[metric], '.2f'))
                    else:
                        rate = float(format(float(events[rate_event]) / 1000000 / rt[metric], '.2f'))
                except:
                    rate = 'n/a'
                derive_events[event_name] = rate

                del events[rate_event]


        #read the rest
        for event_key,event_value in events.items():
            derive_events[event_key] = OrderedDict()
            derive_events[event_key] = event_value

        json_object[region_key] = derive_events.copy()

    return json_object

def sum_json_object(data, derived = False):
    """!
    Function definition for sum_json_object.

    Converts the user supplied .json file containing measurements from PAPI HL 
    function calls to summary format.

    @param data A dictionary containing ranks, threads, and regions.
    @param derived Type of notation. If set to true then the notation is derived.

    @returns An ordered dictionary containing measurements for recorded events. 
             E.g. PAPI_TOT_INS.
    """

    sum_cnt = Sum_Counters()
    for rank, rank_value in data['ranks'].items():
        for thread, thread_value in rank_value['threads'].items():
            for region_value in thread_value['regions'].values():
                sum_cnt.add_region(rank, thread, region_value)

    if derived == True:
        return derive_sum_json_object(sum_cnt.get_json())
    else:
        return sum_cnt.get_json()

def get_ipc_dict(inst, cyc):
    """!
    Function definition for get_ipc_dict.

    Calculates IPC.

    @param inst An ordered dictionary containing measurements for PAPI_TOT_INS.
    @param cyc An ordered dictionary containing measurements for PAPI_TOT_CYC.

    @returns An ordered dictionary containing IPC measurement(s).
    """

    ipc_dict = OrderedDict()
    for (inst_key,inst_value), (cyc_key,cyc_value) in zip(inst.items(), cyc.items()):
        try:
            ipc = float(int(inst_value) / int(cyc_value))
        except:
            ipc = 0
        ipc_dict[inst_key] = float(format(ipc, '.2f'))
    return ipc_dict

def get_ops_dict(ops, rt):
    """!
    Function definition for get_ops_dict.

    Calculates OPS.

    @param ops An ordered dictionary containing measurements for rate recorded 
               events. E.g. PAPI_FP_INS.
    @param rt  An ordered dictionary containing measurements for real time.
               E.g. real_time_nsec.

    @returns An ordered dictionary containing OPS measurement(s).
    """
    ops_dict = OrderedDict()
    for (ops_key,ops_value), (rt_key,rt_value) in zip(ops.items(), rt.items()):
        try:
            ops = float(ops_value) / 1000000 / rt_value
        except:
            ops = 0
        ops_dict[ops_key] = float(format(ops, '.2f'))
    return ops_dict

def convert_value(value, event_type = 'Other'):
    """!
    Function definition for convert_value.

    Converts current measurement precision from a recorded event to a new precision.
  
    @param value Measurement from a recorded event. E.g. PAPI_TOT_INS.
    @param event_type Type of event recorded. E.g. cycles or runtime.

    @returns New precision for the event value. Either an int or float.
    """
    if event_type == 'Other':
        result = float(value)
        result = float(format(result, '.2f'))
    elif event_type == 'Cycles':
        result = int(value)
    elif event_type == 'Runtime':
        result = float(value) / 1.e09
        result = float(format(result, '.2f'))
    elif event_type == 'CPUtime':
        result = float(value) / 1.e09
        result = float(format(result, '.2f'))

    return result

def derive_read_events(events, event_type = 'Other'):
    """!
    Function definition for derive_read_events.

    Format derived event values to a specific precision.

    @param events An ordered dictionary filled with measurements from recorded 
                  events. E.g. PAPI_TOT_INS.
    @param event_type Type of event recorded. E.g. cycles or runtime.

    @returns An ordered dictionary with values formatted to a specific precision 
             (int or float).
    """
    format_read_dict = OrderedDict()
    for read_key,read_value in events.items():
        format_read_dict[read_key] = convert_value(read_value, event_type)
    return format_read_dict

def derive_events(events):
    """!
    Function definition for derive_events.

    Parses an ordered dictionary that contains derived events.

    @param events An ordered dictionary filled with measurements from recorded 
                  events. E.g. PAPI_TOT_INS.
  
    @returns An ordered dictionary filled with formatted measurements for derived
             events.
    """
    #keep order as declared
    derive_events = OrderedDict()
    #remember runtime for other metrics like MFLOPS
    rt = 1.0
    rt_dict = OrderedDict()

    #name
    if 'name' in events:
        derive_events['name'] = events['name']
        del events['name']

    #skip parent_region_id and cycles
    if 'parent_region_id' in events or 'cycles' in events:
        del events['parent_region_id']
        del events['cycles']

    #Real Time
    if 'real_time_nsec' in events:
        if isinstance(events['real_time_nsec'],dict):
            for read_key,read_value in events['real_time_nsec'].items():
                rt_dict[read_key] = convert_value(read_value, 'Runtime')
            derive_events[derived_metric_names['real_time_nsec']] = derive_read_events(events['real_time_nsec'],'Runtime')
        else:
            rt = convert_value(events['real_time_nsec'], 'Runtime')
            derive_events[derived_metric_names['real_time_nsec']] = convert_value(events['real_time_nsec'], 'Runtime')
        del events['real_time_nsec']

    #CPU Time
    if 'perf::TASK-CLOCK' in events:
        if isinstance(events['perf::TASK-CLOCK'],dict):
            derive_events[derived_metric_names['perf::TASK-CLOCK']] = derive_read_events(events['perf::TASK-CLOCK'],'CPUtime')
        else:
            derive_events[derived_metric_names['perf::TASK-CLOCK']] = convert_value(events['perf::TASK-CLOCK'], 'CPUtime')
        del events['perf::TASK-CLOCK']

    #PAPI_TOT_INS and PAPI_TOT_CYC to calculate IPC
    if 'PAPI_TOT_INS' in events and 'PAPI_TOT_CYC' in events:
        if isinstance(events['PAPI_TOT_INS'],dict) and isinstance(events['PAPI_TOT_CYC'],dict):
            ipc_dict = get_ipc_dict(events['PAPI_TOT_INS'], events['PAPI_TOT_CYC'])
            derive_events['IPC'] = ipc_dict
        else:
            try:
                ipc = float(int(events['PAPI_TOT_INS']) / int(events['PAPI_TOT_CYC']))
            except:
                ipc = 0
            derive_events['IPC'] = float(format(ipc, '.2f'))

        del events['PAPI_TOT_INS']
        del events['PAPI_TOT_CYC']
  
    #Rates
    global event_rate_names
    for rate_event in event_rate_names:
        if rate_event in events:
            event_name = event_rate_names[rate_event]
            if isinstance(events[rate_event],dict):
                rate_dict = get_ops_dict(events[rate_event], rt_dict)
                derive_events[event_name] = rate_dict
            else:
                try:
                    rate = float(format(float(events[rate_event]) / 1000000 / rt, '.2f'))
                except:
                    rate = 0
                derive_events[event_name] = rate
            del events[rate_event]

    #read the rest
    for event_key,event_value in events.items():
        if isinstance(event_value,dict):
            derive_events[event_key] = derive_read_events(event_value)
        else:
            derive_events[event_key] = convert_value(event_value)

    return derive_events

def derive_json_object(data):
    """!
    Function defintion for derive_json_object.

    Converts the user supplied .json file containing measurements from PAPI HL
    function calls to derived format.

    @param data Data obtained from PAPI HL function calls.

    @returns An ordered dictionary containing values for ranks, threads, regions, name, 
             real time, and IPC in .json format.
    """
    for rank, rank_value in data['ranks'].items():
        for thread, thread_value in rank_value['threads'].items():
            for region, region_value in thread_value['regions'].items():
                data['ranks'][rank]['threads'][thread]['regions'][region] = derive_events(region_value)
    return data

def write_json_file(data, file_name):
    """!
    Function definition for write_json_file.
  
    Write enhanced output to output file.

    @param data Data obtained from PAPI HL function calls.

    @param file_name Output filename. Either papi.json or papi_sum.json.
    """
    with io.open(file_name, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(data,
                          indent=4, sort_keys=False,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))
        print (str_)

def main(format, type, notation, source_dir = None, source_file = None):
    """!
    Function definition for main.

    Contains code to run upon the Python interpreter executing the file.

    @param format User passed output format, e.g. json.
    @param type User passed output type. Either detailed or summary.
    @param notation User passed notation. Either raw or derived.
    @param source_dir Measurement directory of raw data.
    @param source_file Individual .json file containing measurements of raw data.
    """
    if (format == "json"):
        if source_dir != None:
            json = merge_json_files(source_dir)
        else:
            json = parse_source_file(source_file)

        if type == 'detail':
            if notation == 'derived':
                write_json_file(derive_json_object(json), 'papi.json')
            else:
                write_json_file(json, 'papi.json')

        #summarize data over regions with the same name, threads and ranks
        if type == 'summary':
            if notation == 'derived':
                write_json_file(sum_json_object(json, True), 'papi_sum.json')
            else:
                write_json_file(sum_json_object(json), 'papi_sum.json')
    else:
        print("Format not supported!")

def parse_args():
    """!
    Function definition for parse_args.

    Defines and parses command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, required=False,
                        help='Measurement directory of raw data.')
    parser.add_argument('--source_file', type=str, required=False,
                        help='Individual file containing measurements of raw data.')
    parser.add_argument('--format', type=str, required=False, default='json', 
                        help='Output format, e.g. json.')
    parser.add_argument('--type', type=str, required=False, default='summary', 
                        help='Output type: detail or summary.')
    parser.add_argument('--notation', type=str, required=False, default='derived', 
                        help='Output notation: raw or derived.')
  
    # check to make sure a value has not been passed for both filename and source
    if (parser.parse_args().source_dir != None and
        parser.parse_args().source_file != None):
        # executes if both conditions are true
        print("Cannot pass values to both source_dir and source_file."
              " Value must be passed to either source_dir or source_file.")
        parser.print_help()
        parser.exit()

    # check to see if file exists
    if parser.parse_args().source_file != None:
        source_file = str(parser.parse_args().source_file)
        if not os.path.isfile(source_file):
            print("The file named '{}' does not exist!\n".format(source_file))
            parser.print_help()
            parser.exit()
    # check if papi directory exists
    elif parser.parse_args().source_dir != None:
        source_dir = str(parser.parse_args().source_dir)
        if os.path.isdir(source_dir) == False:
            print("Measurement directory '{}' does not exist!\n".format(source_dir))
            parser.print_help()
            parser.exit()
    # output if neither source_file or source_dir are supplied
    else:
        print("Path to either a JSON file (--source_file) or a" 
              " dictionary (--source_dir) which contains a JSON file is required.")
        parser.print_help()
        parser.exit()
    
    # check format
    output_format = str(parser.parse_args().format)
    if output_format != "json":
        print("Output format '{}' is not supported!\n".format(output_format))
        parser.print_help()
        parser.exit()

    # check type
    output_type = str(parser.parse_args().type)
    if output_type != "detail" and output_type != "summary":
        print("Output type '{}' is not supported!\n".format(output_type))
        parser.print_help()
        parser.exit()

    # check notation
    output_notation = str(parser.parse_args().notation)
    if output_notation != "raw" and output_notation != "derived":
        print("Output notation '{}' is not supported!\n".format(output_notation))
        parser.print_help()
        parser.exit()
  

    return parser.parse_args()

if __name__ == '__main__':
    ##\cond
    args = parse_args()
    main(format=args.format,
         source_dir=args.source_dir,
         source_file=args.source_file,
         type=args.type,
         notation=args.notation)
    ##\endcond
