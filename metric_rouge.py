'''Calculate ROUGE metrics between Abstract pairs in Task 2.'''
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
from rouge_score import rouge_scorer
import numpy as np
import openai
openai.api_key = ""
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# from openai import OpenAI
# client = OpenAI()

def huggingface_api(system_prompt, user_prompt, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    
    API_URL = "https://api-inference.huggingface.co/models/" + model_id
    headers = {"Authorization": f"Bearer hf_key"}
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

def compute_bleu(json_files, model_id):
    
    b1_socres, b2_socres, b3_socres, b4_socres = [], [], [], []
    index = 0 
    for file in json_files:
        print("#########################################" + file + "##############################################" )
        generate = read_json_file(file)
        index += 1
        save_path_nli = os.path.join("abstact_rouge", file)
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
            if not os.path.exists('/'.join(save_path_nli.split('/')[0:-1])):
                os.makedirs('/'.join(save_path_nli.split('/')[0:-1]))
            # "./clean/./llm3/meta-llama_Meta-Llama-3.1-8B-Instruct/data/immunol/Volume 20 (2002)/10.1146_annurev.immunol.20.090501.112049_content.json"
            # print("#########################################" + file + "##############################################" )
            abstract_file = file.replace("./t2/" + model_id, "./clean")
            context = read_json_file(abstract_file)
            abstract1 = context["context"]["ABSTRACT"]
            if "Abstract" in generate.keys():
                abstract2 = generate["Abstract"]
            else:
                abstract2 = ""

            reference = abstract1
            candidate = abstract2

            # 定义平滑函数（用于避免零分数）
            # smoothie = SmoothingFunction().method4

            # 分别计算 BLEU-1, BLEU-2, BLEU-3, BLEU-4
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, candidate)
            # print(scores['rouge1'])
            # exit()
            bleu1 = scores['rouge1'].fmeasure
            bleu2 = scores['rouge2'].fmeasure
            bleu3 = scores['rougeL'].fmeasure
            bleu4 = scores['rougeL'].fmeasure
            


            b1_socres.append(scores['rouge1'].fmeasure)
            b2_socres.append(scores['rouge2'].fmeasure)
            b3_socres.append(scores['rougeL'].fmeasure)
            b4_socres.append(scores['rougeL'].fmeasure)
            results = {"Abstract1": abstract1, "Abstract2": abstract2, "R1": bleu1, "R2": bleu2, "RL": bleu3, "RS": bleu4}

            with open(save_path_nli,'w',encoding='utf-8') as f:
                  json.dump(results, f, ensure_ascii=False)
        else:
            b1_socres.append(results["R1"])
            b2_socres.append(results["R2"])
            b3_socres.append(results["RL"])
            b4_socres.append(results["RS"])
    # print(b1_socres, b2_socres, b3_socres, b4_socres)
    # exit()
    return b1_socres, b2_socres, b3_socres, b4_socres
    




if __name__ == "__main__":
    # llm = "meta-llama_Llama-3.2-3B-Instruct"
    # llm = "gpt-4o-new"
    # llm = "claude"
    # llm = "Qwen_Qwen2.5-72B-Instruct"
    llm = "deepseek"
    model_b1_scores, model_b2_scores, model_b3_scores, model_b4_scores = [],[],[],[]
    topics = [f.name for f in os.scandir("./clean/data") if f.is_dir()]
    for t in topics[0:52]:
        # print(t)
        # exit()
        if t=="arcompsci":
            continue
        
        journal_llm_path = './t2/'+ llm + '/data/' + t

        subdirectories = [d for d in os.listdir(journal_llm_path) if os.path.isdir(os.path.join(journal_llm_path, d))]
        journal_title_score, journal_info_score, journal_halluc_score = [], [], []

        if not os.path.exists(os.path.join("./rouge_results", journal_llm_path)):
            os.makedirs(os.path.join("./rouge_results", journal_llm_path))
        #保存分数
        f = open(os.path.join("./rouge_results", journal_llm_path, t + ".txt"),'w',encoding='utf-8')
        journal_b1_socres, journal_b2_socres, journal_b3_socres , journal_b4_socres  = [], [], [], []


        for subd in subdirectories:
            # if subd == "Volume 13 (2020)":
            if "2023" in subd:
                json_files = find_json_files(os.path.join(journal_llm_path, subd))
                # print(os.path.join(journal_llm_path, subd))
                if not os.path.exists(os.path.join("./clean", journal_llm_path, subd)):
                        os.makedirs(os.path.join("./clean", journal_llm_path, subd))
                # json_files = find_json_files(os.path.join("./clean", journal_llm_path, subd))
                # "NLI result path"
                # if not os.path.exists(os.path.join("./abstract_embdding", journal_llm_path, subd)):
                #         os.makedirs(os.path.join("./abstract_embdding", journal_llm_path, subd))

                # smi_socres = compute_nli_score(json_files, llm)
                # print(json_files)
                b1_socres, b2_socres, b3_socres, b4_socres = compute_bleu(json_files, llm)
                # exit()
                volum_score_b1 = sum(b1_socres)/len(b1_socres)
                volum_score_b2 = sum(b2_socres)/len(b2_socres)
                volum_score_b3 = sum(b3_socres)/len(b3_socres)
                volum_score_b4 = sum(b4_socres)/len(b4_socres)


                journal_b1_socres.append(volum_score_b1)
                journal_b2_socres.append(volum_score_b2)
                journal_b3_socres.append(volum_score_b3)
                journal_b4_socres.append(volum_score_b4)

                f.write("############\n")
                f.write(subd + '\n')
                f.write("volum_score_b1:\t" + str(volum_score_b1) +'\n')
                f.write("volum_score_b2:\t" + str(volum_score_b2) +'\n')               
                f.write("volum_score_b3:\t" + str(volum_score_b3) +'\n')
                f.write("volum_score_b4:\t" + str(volum_score_b4) +'\n')

                print("#########################" + t + " " + subd + "#########################")
                print("volum_score_b1", volum_score_b1)
                print("volum_score_b2", volum_score_b2)
                print("volum_score_b3", volum_score_b3)
                print("volum_score_b4", volum_score_b4)
        journal_b1_score = sum(journal_b1_socres)/len(journal_b1_socres)
        journal_b2_score = sum(journal_b2_socres)/len(journal_b2_socres)
        journal_b3_score = sum(journal_b3_socres)/len(journal_b3_socres)
        journal_b4_score = sum(journal_b4_socres)/len(journal_b4_socres)

        f.write("############\n")
        f.write(t + '\n')
        f.write("journal_b1_score:\t" + str(journal_b1_score) +'\n')
        f.write("journal_b2_score:\t" + str(journal_b2_score) +'\n')
        f.write("journal_b3_score:\t" + str(journal_b3_score) +'\n')
        f.write("journal_b4_score:\t" + str(journal_b4_score) +'\n')

                
        print("#########################" + t  + "#########################")
        print("journal_b1_score", journal_b1_score) 
        print("journal_b2_score", journal_b2_score) 
        print("journal_b3_score", journal_b3_score) 
        print("journal_b4_score", journal_b4_score) 
        model_b1_scores.append(journal_b1_score)
        model_b2_scores.append(journal_b2_score)
        model_b3_scores.append(journal_b3_score)
        model_b4_scores.append(journal_b4_score)
    f1 = open('./rouge_results/t2/' + llm + '.txt','w',encoding='utf-8')
    f1.write("b1" + "\t" + str(sum(model_b1_scores)/len(model_b1_scores))+'\n')
    f1.write("b2" + "\t" + str(sum(model_b2_scores)/len(model_b2_scores))+'\n')
    f1.write("b3" + "\t" + str(sum(model_b3_scores)/len(model_b3_scores))+'\n')
    f1.write("b4" + "\t" + str(sum(model_b4_scores)/len(model_b4_scores))+'\n')
    print(sum(model_b1_scores)/len(model_b1_scores))
    print(sum(model_b2_scores)/len(model_b2_scores))
    print(sum(model_b3_scores)/len(model_b3_scores))
    print(sum(model_b4_scores)/len(model_b4_scores))




        # exit()
                # 

