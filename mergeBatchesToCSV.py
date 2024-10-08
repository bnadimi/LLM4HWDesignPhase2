import psycopg2
import pandas as pd
import numpy as np
import json
import re
import os
import csv
from datasets import load_dataset
from tqdm import tqdm

def remove_non_utf8(content):
    # Keep only ASCII characters (0–127) or UTF-8 characters
    cleaned_content = ''.join(c for c in content if ord(c) < 128)
    return cleaned_content

def extract_numbers(string):
    int_list = []
    for item in re.findall(r'\d+', string):
        int_list.append(int(item))
    return int_list

def read_the_score(json_input):
    json_data = json.loads(json_input)
    score = json_data['response']['body']['choices'][0]['message']['content']
    if extract_numbers(score) == []:
        return (0, extract_numbers(json_data['custom_id'])[0])
    return (extract_numbers(score)[0], extract_numbers(json_data['custom_id'])[0])

def get_UID(json_input):
    json_data = json.loads(json_input)
    return extract_numbers(json_data['custom_id'])[0]

def get_the_code(json_input):
    json_data = json.loads(json_input)
    code = json_data['body']['messages'][0]['content']
    code = "\n".join(code.splitlines()[1:-1])
    return (code, extract_numbers(json_data['custom_id'])[0])

def get_the_code_for_generated(json_input):
    json_data = json.loads(json_input)
    code = json_data['response']['body']['choices'][0]['message']['content']
    code = "\n".join(code.splitlines()[1:-1])
    pattern = re.compile(r'module.*?endmodule', re.DOTALL)
    modules = pattern.findall(code)
    return (modules, extract_numbers(json_data['custom_id'])[0])

def get_description(json_input):
    json_data = json.loads(json_input)
    description = json_data['response']['body']['choices'][0]['message']['content']
    if description == "":
        print("Empty description found at", extract_numbers(json_data['custom_id'])[0])
    return (description, extract_numbers(json_data['custom_id'])[0])

def read_batches(path, file_prefix, get_code_or_score='score'):
    unique_ids = {}
    data = []
    code_or_score = []
    folder_path = path
    file_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
    for i in range(0, file_count):
        file_path_list = path + '/' + file_prefix + str(i) + '.jsonl' 
        with open(file_path_list, 'r') as file:
            for line in file:
                if line == "\n":
                    continue
                UID = get_UID(line)
                if UID not in unique_ids:
                    unique_ids[UID] = 1
                else:
                    unique_ids[UID] = int(unique_ids[UID]) + 1
                    print("Duplicate UID found:", UID)
                data.append(line)
                if get_code_or_score == 'score':
                    code_or_score.append(read_the_score(line))
                elif get_code_or_score == 'code':
                    # code_or_score.append(get_the_code(line))
                    code_or_score.append(get_the_code_for_generated(line))
                else:
                    code_or_score.append(get_description(line))
    for item in unique_ids:
        if unique_ids[item] > 1:
            print("Duplicate UID found:", item)
    print(f"Read {len(data)} data.")
    return data, code_or_score

def read_batches_for_generated(path, file_prefix):
    unique_ids = {}
    data = []
    code = []
    folder_path = path
    file_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
    for i in range(0, file_count-1):
        file_path_list = path + '/' + file_prefix + str(i) + '.jsonl' 
        with open(file_path_list, 'r') as file:
            for line in file:
                if line == "\n":
                    continue
                UID = get_UID(line)
                if UID not in unique_ids:
                    unique_ids[UID] = 1
                else:
                    unique_ids[UID] = int(unique_ids[UID]) + 1
                    print("Duplicate UID found:", UID)
                data.append(line)
                aCode = get_the_code_for_generated(line)
                if len(aCode[0]) != 0:
                    code.append((aCode[0][0], aCode[1]))
    for item in unique_ids:
        if unique_ids[item] > 1:
            print("Duplicate UID found:", item)
    print(f"Read {len(data)} data.")
    return data, code

def mergeAllDatasets(outputFileName, codes, ranks, descriptions):

    with open(outputFileName, 'w', newline='', encoding='utf-8') as csv_file:
        # csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL, escapechar='\\')  
        csv_writer.writerow(["code", "description"])
        for i in range(0, len(codes)):
            code = codes[i][0]
            rank = ranks[i][0] + " out of 20."
            description = descriptions[i][0]
            descriptionDict = {}
            descriptionDict['rank'] = rank
            descriptionDict['description'] = description
            csv_writer.writerow([code, descriptionDict])

def extract_module_header(text):
    module = text.split("Module header:")
    module_header = re.search(r"(module.*?;)", module[1], re.DOTALL) 
    # module_header = re.search(r'(module\s+[a-zA-Z0-9_]+\s*#\(.+?;)', text, re.DOTALL)
    if module_header:
        module_header = module_header.group(1).strip()
        # print("\nModule Header:\n", module_header)
    return str(module_header)





ds = load_dataset("GaTech-EIC/MG-Verilog")
offset = 0
offset2 = 11144
brokenCases = [4049, 5322, 7808]
    
ds_test = ds["train"]
print("Original dataset size: ", len(ds))
print("Original test dataset size: ", len(ds_test))

half_code_samples = []
code_samples = []
block_summary_samples = []
high_level_global_summary_samples = []
detailed_global_summary_samples = []

moduleHeader_block = []
moduleHeader_high = []
moduleHeader_detailed = []

compile_results = []


for i in tqdm(range(offset, offset+offset2)):
    half_code_sample = ds_test[i]['code']

    block_summary_sample = ds_test[i]['description']['block_summary']
    high_level_global_summary_sample = ds_test[i]['description']['high_level_global_summary']
    detailed_global_summary_sample = ds_test[i]['description']['detailed_global_summary']

    header_block = extract_module_header(block_summary_sample)
    header_high = extract_module_header(high_level_global_summary_sample)
    header_detailed = extract_module_header(detailed_global_summary_sample)

    code_sample = header_block + "\n" + half_code_sample

    if header_block == "None" or header_high == "None" or header_detailed == "None":
        print("Module header not found")
        print("block_summary: ", block_summary_samples[i-offset])
        print("header_block: ", header_block)
        print("header_high: ", header_high)
        print("header_detailed: ", header_detailed)
        print(i)
        exit()

    if i in brokenCases:
        # brokenSamples.append(code_sample)
        code_samples.append(code_sample)
        compile_results.append([False, 0, "i give up"])
        continue

        
    half_code_samples.append(half_code_sample)
    block_summary_samples.append(block_summary_sample)
    high_level_global_summary_samples.append(high_level_global_summary_sample)
    detailed_global_summary_samples.append(detailed_global_summary_sample)

    code_samples.append(code_sample)



ranked_data, all_scores                     = read_batches('GPT-Batches/results/rank', 'out_ranks_batch_', 'score')
testBenchesData, all_testBenches            = read_batches('GPT-Batches/results/testbench', 'out_testBenches_batch_', 'code')


print(len(all_scores))
print(len(all_testBenches))


print(f"Length of all_scores: {len(all_scores)}")
print(f"Length of all_testBenches: {len(all_testBenches)}")

sorted_array_testBenches = all_testBenches
sorted_array_scores = all_scores


outputFileName = 'FinalMergedWithRankAndDescriptions.csv'
counter = 0
with open(outputFileName, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL, escapechar='\\')  
    csv_writer.writerow(["code", "description"])

    for i in range(0, len(sorted_array_scores)):
        code = None
        rank        = sorted_array_scores[i][0]
        if len(sorted_array_testBenches[i][0]) == 0:
            print("Empty testbench found at", sorted_array_scores[i][1])
            testBench = ""
        else:
            testBench = sorted_array_testBenches[i][0][0]
        descriptionDict = {
            'rank': str(rank),
            'testBench': str(testBench)
        }
        json_string = json.dumps(descriptionDict)
        csv_writer.writerow([code, json_string])
        counter += 1
        
print("Total number of entries found: ", counter)

            
