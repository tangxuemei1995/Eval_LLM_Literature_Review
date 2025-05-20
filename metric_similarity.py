'''
Task 2 evaluation protocol:

Encode original and generated abstracts using text-embedding-3-large

Compute semantic similarity between embedding vectors

Evaluate abstract quality based on cosine similarity metrics
'''
from google_api import find_google_scholar
from llm_api import find_json_files, read_json_file
import os
import json
import re
import time
import csv
from bing_api import query_bing_return_mkt
from google_search_api import google_search
from semantic_scholar import semantic_scholar_search
from clean_llm_data import clean_references_for_second
# from evaluate import gpt
import numpy as np
import openai
openai.api_key = ""


def huggingface_api(system_prompt, user_prompt, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    
    API_URL = "https://api-inference.huggingface.co/models/" + model_id
    headers = {"Authorization": f"Bearer key"}
    data = {"inputs": system_prompt + user_prompt}
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()[0]["generated_text"]


def gpt_3(system_prompt, user_prompt):
    ''' 调用gpt3.5 0.006$/1K token'''
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    max_tokens=3700)  
    result = completion["choices"][0][ "message"]["content"]
    return result


def gpt_4o(system_prompt, user_prompt):
    ''' 调用gpt3.5 0.006$/1K token'''
    completion = openai.ChatCompletion.create(
        model="gpt-4o", 
        messages=[{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}],
        temperature=0.0,
        # response_format={"type": "json_object"},
    max_tokens=3700)  
    result = completion["choices"][0][ "message"]["content"]
    return result

f = open('./prompt3-gpt.txt', 'r', encoding='utf-8')
# f.read()
system_prompt = f.read().strip()

f = open('./prompts/prompt_nli.txt', 'r', encoding='utf-8')
# f.read()
system_prompt_nli = f.read().strip()
             

def cosine_similarity(vec1, vec2):
    # 计算两个向量的余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

def compute_similarity(json_files, model_id):
    
    smi_socres = []
    index = 0 
    for file in json_files:
        print("#########################################" + file + "##############################################" )
        generate = read_json_file(file)
        index += 1
        save_path_nli = os.path.join("abstract_embdding_large", file)
        results = {}
        # print(save_path_nli)
        
        if os.path.exists(save_path_nli):
            with open(save_path_nli, 'r', encoding='utf-8') as f:
                results = json.load(f)
        # else: 
        #     results = {}
        # print("results",results)
        if results == {}:
            '''
            save path for the best candidate
            '''
            # save_path = file.replace('./clean','./nli_data')
            # if not os.path.exists('/'.join(save_path.split('/')[0:-1])):
            #     os.makedirs('/'.join(save_path.split('/')[0:-1]))
            # "./clean/./llm3/meta-llama_Meta-Llama-3.1-8B-Instruct/data/immunol/Volume 20 (2002)/10.1146_annurev.immunol.20.090501.112049_content.json"
            # print("#########################################" + file + "##############################################" )
            # abstract_file = file.replace("./t2/" + model_id, "./clean")
            # context = read_json_file(abstract_file)
            # abstract1 = context["context"]["ABSTRACT"]
            


            # ab1_embed = read_json_file(os.path.join("abstract_embdding_large", file.replace(model_id,"gpt-4o")))
            ab1_embed = read_json_file(os.path.join("abstract_embdding_large", file.replace(model_id,"Qwen_Qwen2.5-72B-Instruct")))
            abstract1_embd =  ab1_embed["Abstract1"]
            # print(ab1_embed)
            # exit()
            if "Abstract" in generate.keys():
                abstract2 = generate["Abstract"]
            else:
                abstract2 = ""

            
            # response = openai.Embedding.create(
            #     input=abstract1,
            #     model="text-embedding-3-large"
            # )

            # abstract1_embd = response.data[0].embedding

            response = openai.Embedding.create(
                input=abstract2,
                model="text-embedding-3-large"
            )

            abstract2_embd = response.data[0].embedding
            sim = cosine_similarity(abstract1_embd, abstract2_embd)
            smi_socres.append(sim)
            results = {"Abstract1": abstract1_embd, "Abstract2": abstract2_embd, "Sim": sim}

            with open(save_path_nli,'w',encoding='utf-8') as f:
                  json.dump(results, f, ensure_ascii=False)
        else:
            smi_socres.append(results["Sim"])
    return smi_socres
    




if __name__ == "__main__":
    # llm = "meta-llama_Llama-3.2-3B-Instruct"
    llm = "deepseek"
    # llm = "gpt-4o-new"
    # llm = "claude"
    # llm = "Qwen_Qwen2.5-72B-Instruct"
    topics = [f.name for f in os.scandir("./clean/data") if f.is_dir()]
    model_similarity = []
    for t in topics[0:52]:
        # print(t)
        # exit()
        if t=="arcompsci":
            continue
        
        journal_llm_path = './t2/'+ llm + '/data/' + t

        subdirectories = [d for d in os.listdir(journal_llm_path) if os.path.isdir(os.path.join(journal_llm_path, d))]
        journal_title_score, journal_info_score, journal_halluc_score = [], [], []

        if not os.path.exists(os.path.join("./sim_results_large", journal_llm_path)):
            os.makedirs(os.path.join("./sim_results_large", journal_llm_path))
    

        #保存分数
        f = open(os.path.join("./sim_results_large", journal_llm_path, t + ".txt"),'w',encoding='utf-8')
        
        journal_smi_socres = []

        for subd in subdirectories:
            # if subd == "Volume 13 (2020)":
            if "2023" in subd:
                json_files = find_json_files(os.path.join(journal_llm_path, subd))
                # print(os.path.join(journal_llm_path, subd))
                if not os.path.exists(os.path.join("./clean", journal_llm_path, subd)):
                        os.makedirs(os.path.join("./clean", journal_llm_path, subd))
                # json_files = find_json_files(os.path.join("./clean", journal_llm_path, subd))
                "NLI result path"
                if not os.path.exists(os.path.join("./abstract_embdding_large", journal_llm_path, subd)):
                        os.makedirs(os.path.join("./abstract_embdding_large", journal_llm_path, subd))

                # smi_socres = compute_nli_score(json_files, llm)
                # print(json_files)
                smi_socres = compute_similarity(json_files, llm)
                # exit()
                volum_score = sum(smi_socres)/len(smi_socres)
                journal_smi_socres.append(volum_score)
                f.write("############\n")
                f.write(subd + '\n')
                f.write("volum_score:\t" + str(volum_score) +'\n')
                
                print("#########################" + t + " " + subd + "#########################")
                print("volum_score", volum_score)
        journal_nli_score = sum(journal_smi_socres)/len(journal_smi_socres)
        f.write("############\n")
        f.write(t + '\n')
        f.write("journal_nli_score:\t" + str(journal_nli_score) +'\n')
                
        print("#########################" + t  + "#########################")
        print("journal_nli_score", journal_nli_score) 
        model_similarity.append(journal_nli_score)
    model_score = sum(model_similarity)/len(model_similarity)
    print(llm, model_score)

        
                 

