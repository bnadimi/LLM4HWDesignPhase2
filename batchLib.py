from openai import OpenAI
import json
import csv
import tiktoken

class GPTBatchCommands:
    def __init__(self):
        self.client = OpenAI(
        api_key = [your_key]
    )   
        
    def submitBatch(self, batch_file_list, purpose):  
        list_of_batch_ids = []
        batchCounter = 0
        for a_batch_file in batch_file_list:
            batch_input_file = self.client.files.create(file=open(a_batch_file, "rb"), purpose="batch")

            batch_input_file_id = batch_input_file.id

            self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": purpose
                }
            )
            aJob = self.client.batches.list()
            print("Job type: ", type(aJob))
            counter = 0
            for job in aJob:
                counter += 1
                list_of_batch_ids.append(job.id)
                break
            batchCounter += 1
            # print("Batch submitted.")

        print("Number of batches submitted: ", batchCounter)
        print("List of batch IDs: ", list_of_batch_ids)

    def getBatchStatus(self, batch_id_list):
        for batch_id in batch_id_list:
            # print(client.batches.retrieve(batch_id))
            print(self.client.batches.retrieve(batch_id).status)
            print(self.client.batches.retrieve(batch_id).request_counts)
            print("")
    
    def getOutputFilenames(self, batch_id_list):
        output_filenames = []
        for batch_id in batch_id_list:
            output_filenames.append(self.client.batches.retrieve(batch_id).output_file_id)
        print("Output filenames: ", output_filenames)
        return output_filenames

    def retrieveBatchResults(self, batch_id_list, filename_list):
        for i in range(len(batch_id_list)):
            file_response = self.client.files.content(batch_id_list[i])
            # print("File response: ", file_response.read().decode('utf-8'))
            binary_content = file_response.read() 
            decoded_content = binary_content.decode('utf-8') 
            listResponse = decoded_content.split("\n")
            print("Len response: ", len(listResponse))  
            with open(filename_list[i], 'w') as file:
                for entry in listResponse:
                    file.write(entry) 
                    file.write('\n')

    def cancelBatch(self,  batch_id_list):
        for batch_id in batch_id_list:
            self.client.batches.cancel(batch_id)
            print("Batch cancelled.")

    def getListofJobs(self):
        listOfJobs = self.client.batches.list(limit=1000)
        return listOfJobs
    
    def printListofJobs(self):
        listOfJobs = self.client.batches.list(limit=100)
        for job in listOfJobs:
          print(job.id, job.status, job.output_file_id)


class batchCreator():   
    def __init__(self):
        self.file_counter = 0
        self.row_limit = 45000
        self.max_tokens_per_batch = 10000000  # 20 million token limit
        self.current_batch_tokens = 0
        self.current_batch_rows = 0
        pass

    def estimate_tokens(self, aRequest, model_name="gpt-4o-mini"):
        return len(tiktoken.encoding_for_model(model_name).encode(aRequest)) 
    
    def createSingleRequestForRanking(self, inputPrompt, requestNumber, model_name="gpt-4o-mini", rankOrDescription="rank", temperature=0.5):
        content_rank = "Act as a teacher and rank the quality of this Verilog code in scale of 1 to 20, with 1 being syntactically incorrect and 20 being very efficient and good Verilog code:\n"
        content_rank += inputPrompt  
        content_rank += "\nJust give me the score only."

        content_description = "just give me a functionality summary of what this Verilog code does: \n"
        content_description += inputPrompt
        content_description += "\nShort answer please."

        content_code_generation = "Please act as a professional Verilog designer. Give me the Verilog code for the following instruction: "
        content_code_generation += inputPrompt
        content_code_generation += " Just give me the verilog code."

        final_content = content_rank if rankOrDescription == "rank" else content_description if rankOrDescription == "description" else content_code_generation


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
                    # {
                    #     "role": "user",
                    #     "content": testContent
                    # },
                ],
                "max_tokens": 4000,
                "temperature": temperature
            }
        }
        return jsonl_entry
    
    def createBatch(self, codeList, destinationAddress, model_name="gpt-4o-mini", rankOrDescription="rank", tempList=None):

        max_token_found = 0 
        file_counter = 0
        current_batch_rows = 0
        current_batch_tokens = 0
        tokens_for_this_request = 0
        jsonl_final_file_name = f"{destinationAddress}/BatchNumber_{file_counter}.jsonl"
        jsonl_file = open(jsonl_final_file_name, mode='w')
        
        for i, aCode in enumerate(codeList):

            aRequest = self.createSingleRequestForRanking(aCode, i, model_name, rankOrDescription, temperature=tempList[i] if tempList else 0.5)
            tokens_for_this_request = self.estimate_tokens(aRequest["body"]["messages"][0]["content"], model_name)
            if tokens_for_this_request > max_token_found:
                max_token_found = tokens_for_this_request
                # print("Max token found: ", max_token_found)
            # print("number of tokens: ", self.estimate_tokens(aRequest["body"]["messages"][0]["content"] , model_name))

            if current_batch_rows >= self.row_limit or current_batch_tokens + tokens_for_this_request > self.max_tokens_per_batch:
                # Close the current file and start a new one
                jsonl_file.close()
                file_counter += 1
                jsonl_final_file_name = f"{destinationAddress}/BatchNumber_{file_counter}.jsonl"
                jsonl_file = open(jsonl_final_file_name, mode='w')
                print("Current batch tokens:", current_batch_tokens)
                print("Current batch rows:", current_batch_rows)
                print()
                current_batch_tokens = 0  # Reset token count for new batch
                current_batch_rows = 0    # Reset row count for new batch

            # Write the entry to the JSONL file
            jsonl_file.write(json.dumps(aRequest) + '\n')

            # Update the token and row count
            current_batch_tokens += tokens_for_this_request
            current_batch_rows += 1

        print("Max token found: ", max_token_found)
        print("Current batch tokens:", current_batch_tokens)
        print("Current batch rows:", current_batch_rows)
        print()
        # Close the last file after processing all rows
        jsonl_file.close()
        print("Batch creation completed.")
    
    def getTotalTokens(self, codeList, model_name="gpt-4o-mini"):
        total_request_tokens = 0
        total_code_tokens = 0
        max_token_found = 0
        for i, aCode in enumerate(codeList):
            aRequest = self.createSingleRequestForRanking(aCode[1], i, model_name)
            aRequestSize = self.estimate_tokens(aRequest["body"]["messages"][0]["content"], model_name)
            total_request_tokens += aRequestSize
            total_code_tokens += self.estimate_tokens(aCode[1], model_name)
            if aRequestSize > max_token_found:
                max_token_found = aRequestSize
            # if i % 1000 == 0:
            #     print("Total request tokens: ", total_request_tokens)
            #     print("Total code tokens: ", total_code_tokens)
        return total_request_tokens, total_code_tokens, max_token_found
    
    def selectOnesWithLessThan20000Tokens(self, codeList, model_name="gpt-4o-mini"):
        selected_codes = []
        for i, aCode in enumerate(codeList):
            if self.estimate_tokens(aCode[1], model_name) < 20000:
                selected_codes.append(aCode)
        return selected_codes
        

