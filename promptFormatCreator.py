from datasets import load_dataset
import tiktoken
import json
import re
import subprocess
import os
from tqdm import tqdm
from batchLib import GPTBatchCommands
import csv

class PromptFormatCreator:
    def __init__(self, input_data, max_tokens_per_batch=4096):
        self.input_data = input_data
        self.max_tokens_per_batch = max_tokens_per_batch

    def saveToJSONL(self, destination_address, data):
        jsonl_final_file_name = f"{destination_address}/batch_{0}.jsonl"
        jsonl_file = open(jsonl_final_file_name, mode='w')
        for i, prompt in enumerate(data):
            jsonl_file.write(json.dumps(prompt) + '\n')
        jsonl_file.close()

    def estimate_tokens(self, aRequest, model_name="gpt-4o-mini"):
        return len(tiktoken.encoding_for_model(model_name).encode(aRequest)) 

    def createSinglePrompt(self, inputPrompt, requestNumber, model_name, temperature=0.5, rankOrDescription="rank"):
        content_rank = "Act as a teacher and rank the quality of this Verilog code in scale of 1 to 20, with 1 being syntactically incorrect and 20 being very efficient and good Verilog code:\n"
        content_rank += inputPrompt  
        content_rank += "\nJust give me the score only."

        content_description = "just give me a functionality summary of what this Verilog code does: \n"
        content_description += inputPrompt
        content_description += "\nShort answer please."

        content_code_generation = "Please act as a professional Verilog designer. Give me the Verilog code for the following instruction: "
        content_code_generation += inputPrompt
        content_code_generation += " Just give me the verilog code."

        content_create_test_bench = "Please act as a professional Verilog designer. Create a test bench for the following code: "
        content_create_test_bench += inputPrompt
        content_create_test_bench += " Just give me the verilog test bench code."

        final_content = content_rank if rankOrDescription == "rank" else content_description if rankOrDescription == "description" else content_code_generation if rankOrDescription=="codeGeneration" else content_create_test_bench

        jsonl_entry = {
            "custom_id": f"request-{requestNumber}",
            "method": "POST",
            "url": "/v1/chat/completions",
             "body": {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": final_content
                    }
                ],
                "max_tokens": 4000,
                "temperature": temperature
            }
        }
        return jsonl_entry

    def createBatchRequests(self, codeList, destinationAddress, model_name, rankOrDescription, temperature=0.5):
        jsonl_final_file_name = f"{destinationAddress}/BatchNumber_{0}.jsonl"
        jsonl_file = open(jsonl_final_file_name, mode='w')
        max_token_found = 0

        for i, aCode in enumerate(codeList):
            aRequest = self.createSinglePrompt(aCode, i, model_name, temperature, rankOrDescription)
            tokens_for_this_request = self.estimate_tokens(aRequest["body"]["messages"][0]["content"], model_name)
            if tokens_for_this_request > max_token_found:
                max_token_found = tokens_for_this_request

            jsonl_file.write(json.dumps(aRequest) + '\n')
        jsonl_file.close()
        print("Batch creation completed.")

# Extract Module header
def extract_module_header(text):
    module = text.split("Module header:")
    module_header = re.search(r"(module.*?;)", module[1], re.DOTALL) 
    if module_header:
        module_header = module_header.group(1).strip()
    return str(module_header)

def check_compile(code):
    vvpFlag = True

    open('singleCheck.v', 'w').write(code)
    result = subprocess.run(['iverilog', '-o', 'singleCheck.vvp', 'singleCheck.v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return_code = result.returncode
    output = result.stdout
    errors = result.stderr

    if not os.path.exists("singleCheck.vvp"):
        vvpFlag = False
    else:
        os.remove("singleCheck.vvp")

    os.remove("singleCheck.v")
    return [vvpFlag, return_code, errors]

def readCSVFile(inputFileName):
    input_file_name = inputFileName
    readCSV = []

    with open(input_file_name, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            code = row['code']
            description_dict = json.loads(row['description'])
            readCSV.append([code, description_dict])

    return readCSV

def getMG_VerilogDataset():
    readCSV = readCSVFile("FinalMergedWithRankAndDescriptions.csv")

    ds = load_dataset("GaTech-EIC/MG-Verilog")
    startPoint = 0
    offset = len(ds["train"]) #11144
    brokenCases = [4049, 5322, 7808]
    brokenSamples = []
    
    ds_train = ds["train"]
    print("Original dataset size: ", len(ds))
    print("Original test dataset size: ", len(ds_train))

    half_code_samples = []
    code_samples = []
    block_summary_samples = []
    high_level_global_summary_samples = []
    detailed_global_summary_samples = []

    moduleHeader_block = []
    moduleHeader_high = []
    moduleHeader_detailed = []

    compile_results = []

    counterFine = 0
    counterFailed = 0
    giveUpCounter = 0

    for i in tqdm(range(startPoint, startPoint+offset)):
        half_code_sample = ds_train[i]['code']


        block_summary_sample             = ds_train[i]['description']['block_summary']
        high_level_global_summary_sample = ds_train[i]['description']['high_level_global_summary']
        detailed_global_summary_sample   = ds_train[i]['description']['detailed_global_summary']

        header_block = extract_module_header(block_summary_sample)
        header_high = extract_module_header(high_level_global_summary_sample)
        header_detailed = extract_module_header(detailed_global_summary_sample)

        code_sample = header_block + "\n" + half_code_sample

        if header_block == "None" or header_high == "None" or header_detailed == "None":
            print("Module header not found")
            print("block_summary: ", block_summary_sample)
            print("header_block: ", header_block)
            print("header_high: ", header_high)
            print("header_detailed: ", header_detailed)
            print(i)
            exit()
        if header_block != header_high or header_block != header_detailed or header_high != header_detailed:
            print("Module header not equal")
            exit()

        if i in brokenCases:
            brokenSamples.append(code_sample)
            code_samples.append(code_sample)
            half_code_samples.append(half_code_sample)
            block_summary_samples.append(block_summary_sample)
            high_level_global_summary_samples.append(high_level_global_summary_sample)
            detailed_global_summary_samples.append(detailed_global_summary_sample)

            compile_results.append([False, 0, "i give up"])
            giveUpCounter += 1
            continue

        
        half_code_samples.append(half_code_sample)
        block_summary_samples.append(block_summary_sample)
        high_level_global_summary_samples.append(high_level_global_summary_sample)
        detailed_global_summary_samples.append(detailed_global_summary_sample)
        moduleHeader_block.append(header_block)
        moduleHeader_high.append(header_high)
        moduleHeader_detailed.append(header_detailed)

        code_samples.append(code_sample)
        compile_results.append(check_compile(code_sample))

        if compile_results[i][0] == False:
            if "give up" in compile_results[i][2]:
                giveUpCounter += 1
            else:
                counterFailed += 1
        else:
            counterFine += 1

    print("Number of all code samples: ", len(compile_results))
    print("Number of code samples that compiled successfully: ", counterFine)
    print("Number of code samples that failed to compile: ", counterFailed)
    print("Number of code samples that gave up: ", giveUpCounter)
    exit()


    outputFileName = 'FinalAllMG-Verilog.csv'
    counter = 0
    with open(outputFileName, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL, escapechar='\\')  
        csv_writer.writerow(["code", "description"])

        for i in range(0, len(code_samples)):
            code = half_code_samples[i]
            rank = readCSV[i][1]['rank']
            testBench = readCSV[i][1]['testBench']
            if compile_results[i-startPoint][0] == False:
                if "give up" in compile_results[i-startPoint][2]:
                    compileNote = "This Verilog code contains syntax error!"
                else:
                    compileNote = "This Verilog code has dependency issues!"
            else:
                compileNote = "This Verilog code could compile successfuly with iVerilog!"
            block_summary = block_summary_samples[i]
            high_level_global_summary = high_level_global_summary_samples[i]
            detailed_global_summary = detailed_global_summary_samples[i]
            
            descriptionDict = {
                'rank': str(rank),
                'testBench': str(testBench),
                'compileNote': str(compileNote),
                'block_summary': str(block_summary),
                'detailed_global_summary': str(detailed_global_summary),
                'high_level_global_summary': str(high_level_global_summary)
            }
            json_string = json.dumps(descriptionDict)
            csv_writer.writerow([code, json_string])
            counter += 1

            
    print(brokenSamples)
    return code_samples, block_summary_samples, high_level_global_summary_samples, detailed_global_summary_samples, compile_results


def __main__():
    codeSamples, blockSummarySamples, highLevelGlobalSummarySamples, detailedGlobalSummarySamples, compileResults = getMG_VerilogDataset() 
    # PromptFormatCreatorInstance = PromptFormatCreator(codeSamples)
    # PromptFormatCreatorInstance.createBatchRequests(codeSamples, "GPT-Batches/requests", "gpt-4o-mini", "rank", 0.5)
    # PromptFormatCreatorInstance.createBatchRequests(codeSamples, "GPT-Batches/requests", "gpt-4o-mini", "createTestBench", 0.5)
    # codeSamples, blockSummarySamples, highLevelGlobalSummarySamples, detailedGlobalSummarySamples, compileResults = getMG_VerilogDataset()

def gpt():
    client = GPTBatchCommands()
    # # For submitting the batch files for Generating Descriptions for the ones that don't have
    # batch_file_list = []
    # for i in range(0, 1):
    #     # file_path = f"GPT-Batches/requests/rankingBatches_{i}.jsonl"   
    #     file_path = f"GPT-Batches/requests/testBenchBatches_{i}.jsonl"   
    #     batch_file_list.append(file_path)
    # # client.submitBatch(batch_file_list, "RankingVerilogCodes")
    # client.submitBatch(batch_file_list, "creatingTestBenches")

    # client.cancelBatch(['batch_66ff01c990748190870b78bd9413cd66'])
    # client.printListofJobs()

    # list_of_batches = ['batch_66ff01363d34819085e70ef886190aa4'] # For Ranking
    # list_of_batches = ['batch_66ff018f966c81909dc274e30a164124'] # For Creating Test Benches
    list_of_batches = ['batch_66ff01363d34819085e70ef886190aa4', 'batch_66ff018f966c81909dc274e30a164124'] # All together
    client.getBatchStatus(list_of_batches)
    client.getOutputFilenames(list_of_batches)

    # Output_filenames = ['file-hSUPTt63Td1LbgLW9lxjhuJu']  # Test Benches
    # Output_filenames = ['file-zsfGyA0ZA0Qwv1clEMa950Uz']  
    # for i in range(0, 1):
    #     print(Output_filenames[i])
    #     fileNames = [Output_filenames[i]]
    #     client.retrieveBatchResults(fileNames, [f"GPT-Batches/results/out_ranks_batch_{i}.jsonl"])


if __name__ == "__main__":
    # gpt()
    __main__()