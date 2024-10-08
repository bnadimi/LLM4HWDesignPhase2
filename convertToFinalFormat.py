import json
import pandas as pd
from batchLib import batchCreator
        
ds = pd.read_csv('FinalAllMG-Verilog.csv')

jsonl_final_file_name = "ConvertedFormatMG-VerilogPhase2.jsonl"
jsonl_file = open(jsonl_final_file_name, mode='w')

for i in range(len(ds)):
    
    description_dict = json.loads(ds['description'][i])
    outputValue = ds['code'][i]
    inputRank                   = description_dict['rank']
    inputTestBench              = description_dict['testBench']
    inputCompileNote            = description_dict['compileNote']
    inputBlockSummary           = description_dict['block_summary']
    inputDetailedGlobalSummary  = description_dict['detailed_global_summary']
    inputHighLevelGlobalSummary = description_dict['high_level_global_summary']

    inputString = "block_summary: \n" + inputBlockSummary + "\n detailed_global_summary \n" + inputDetailedGlobalSummary + "\n high_level_global_summary \n" + inputHighLevelGlobalSummary + "\n Ranking (out of 20): \n" + inputRank + "\n test_bench: \n" + inputTestBench + "\n compile_note: \n" + inputCompileNote

    outputString = outputValue
    dict = {"input": inputString, "output": outputString}
    jsonl_file.write(json.dumps(dict) + '\n')

    if i % 1000 == 0:
        print(i)
        