'''Calculate entailment relationships between summaries in Task 2 using GPT-4o as the computation tool.'''
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
# import litellm
import openai, requests

openai.api_key = ""



def huggingface_api(system_prompt, user_prompt, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    
    API_URL = "https://api-inference.huggingface.co/models/" + model_id
    headers = {"Authorization": f"Bearer hf key"}
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

def compute_nli_score(json_files, model_id):
    
    nli_scores = []
    index = 0 
    for file in json_files:
        print("#########################################" + file + "##############################################" )
        generate = read_json_file(file)
        index += 1
        save_path_nli = file.replace("./clean/.","NLI_result")
        results = {}
        if os.path.exists(save_path_nli):
            with open(save_path_nli, 'r', encoding='utf-8') as f:
                results = json.load(f)
        # else: 
        #     results = {}

        if results == {}:
            '''
            save path for the best candidate
            '''
            # save_path = file.replace('./clean','./nli_data')
            # if not os.path.exists('/'.join(save_path.split('/')[0:-1])):
            #     os.makedirs('/'.join(save_path.split('/')[0:-1]))
            # "./clean/./llm3/meta-llama_Meta-Llama-3.1-8B-Instruct/data/immunol/Volume 20 (2002)/10.1146_annurev.immunol.20.090501.112049_content.json"
            # print("#########################################" + file + "##############################################" )
            abstract_file = file.replace("./llm3/" + model_id + "/","")
            abstract = read_json_file(abstract_file)
            abstract = abstract["context"]["ABSTRACT"]

            # exit()
            if "Literature Review" in generate.keys():
                if isinstance(generate["Literature Review"], dict):
                    if "References" in generate["Literature Review"].keys():
                        t, r = "", generate["Literature Review"]["References"]
                else:
                    t, r = generate["Literature Review"], generate["References"]
            else:
                t = ""
                r =  generate["References"]

            if  t == "":
                '''如果有写结果中没有literature,则重新调用gpt-3.5,如果评价其他模型则需要更换这部分'''
                user_prompt = "ABSTRACT:" + abstract

                save_path = file.replace(model_id, model_id.replace('/', '_')+"_supply")
                if not os.path.exists('/'.join(save_path.split('/')[0:-1])):
                    os.makedirs('/'.join(save_path.split('/')[0:-1]))

                if os.path.exists(save_path):

                    t = read_json_file(save_path)
                    if "Literature Review" in t.keys():
                        t = t["Literature Review"]
                    else:
                        t = ""

                if  t == "":
                    try:
                        if not model_id.startswith('gpt'):
                            result = huggingface_api(system_prompt, user_prompt,model_id=model_id.replace("_","/"))
                            llm_gene_path = file.replace('./clean/','')
                            with open(llm_gene_path, 'w', encoding='utf-8') as f4:
                                    json.dump(result, f4, ensure_ascii=False)
                            result = clean_references_for_second([llm_gene_path], model_id=model_id.replace("_","/"))
                        else:
                            result = gpt_4o(system_prompt, user_prompt)

                        # result = gpt_3(system_prompt, user_prompt)
                        
                        result = result
                        with open(save_path, 'w', encoding='utf-8') as f4:
                            json.dump(result, f4, ensure_ascii=False)
                        t = result["Literature Review"]
                    except:
                        try:
                            if not model_id.startswith('gpt'):
                                result = huggingface_api(system_prompt, user_prompt,model_id=model_id.replace("_","/"))
                                llm_gene_path = file.replace('./clean/','')
                                with open(llm_gene_path, 'w', encoding='utf-8') as f4:
                                    json.dump(result, f4, ensure_ascii=False)
                                result = clean_references_for_second([llm_gene_path], model_id=model_id.replace("_","/"))
                            else:
                                result = gpt_4o(system_prompt, user_prompt)

                            # result = gpt_3(system_prompt, user_prompt)
                            # result = json.loads(str(result))
                            with open(save_path, 'w', encoding='utf-8') as f4:
                                json.dump(result, f4, ensure_ascii=False)
                            t = result["Literature Review"]
                        except:
                            # result = gpt_3(system_prompt, user_prompt)
                            if not model_id.startswith('gpt'):
                                result = huggingface_api(system_prompt, user_prompt,model_id=model_id.replace("_","/"))
                                llm_gene_path = file.replace('./clean/','')
                                with open(llm_gene_path, 'w', encoding='utf-8') as f4:
                                    json.dump(result, f4, ensure_ascii=False)
                                result = clean_references_for_second([llm_gene_path], model_id=model_id.replace("_","/"))
                            else:
                                result = gpt_4o(system_prompt, user_prompt)
                            

                            # result = json.loads(str(result))
                            with open(save_path, 'w', encoding='utf-8') as f4:
                                json.dump(result, f4, ensure_ascii=False)
                            t = result["Literature Review"]
                         
            print("abstract", abstract)
            print("literature review:", t)
            print(len(system_prompt_nli) +len(abstract) +len(t))
            response = gpt_4o(system_prompt_nli, "abstract:\n" + abstract +'\n' + "Lterature review" + t)
            response = response.lower()
            

            results = {"Abstract":abstract, "Lterature review":t, "Response":response}
            with open(save_path_nli,'w',encoding='utf-8') as f:
                  json.dump(results, f, ensure_ascii=False)
                 
        res = results["Response"]
        if "entailment" in res:
            nli_scores.append(1)
        else:
            nli_scores.append(0) 
    return nli_scores
    
             

def compute_nli_abstract_score(json_files, model_id):
    
    entail_scores = []
    contra_scores = []
    neutr_scores = []
    index = 0 
    for file in json_files:
        print("#########################################" + file + "##############################################" )
        generate = read_json_file(file)
        index += 1
        save_path_nli =os.path.join("NLI_result_new", file)
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
            abstract_file = file.replace("./t2/" + model_id,"./clean")
            context = read_json_file(abstract_file)
            abstract1 = context["context"]["ABSTRACT"]
            # print(generate)
            # exit()
            if "Abstract" in generate.keys():
                abstract2 = generate["Abstract"]
            else:
                abstract2 = ""

            print("abstract1", abstract1)
            print("abstract2", abstract2)
            print(len(system_prompt_nli) + len(abstract1) +len(t))
            response = gpt_4o(system_prompt_nli, "Abstract1:\n" + abstract1 +'\n' + "Abstract2" + abstract2)
            response = response.lower()
            results = {"Abstract1": abstract1, "Abstract2": abstract2, "Response":response}
            with open(save_path_nli,'w',encoding='utf-8') as f:
                  json.dump(results, f, ensure_ascii=False)
        res = results["Response"]
        if "entailment" in res:
            entail_scores.append(1)
        elif "contradiction" in res:
            contra_scores.append(0) 
        elif "neutrality"  in res:
            neutr_scores.append(0)
        else:
            print("response:", res)
            exit()
    return entail_scores, contra_scores, neutr_scores
    

if __name__ == "__main__":
    # llm = "gpt-4o-new"
    # llm = "claude"
    # llm = "Qwen_Qwen2.5-72B-Instruct"
    # llm = "meta-llama_Llama-3.2-3B-Instruct"\
    llm = "deepseek"

    topics = [f.name for f in os.scandir("./clean/data") if f.is_dir()]
    model_nli = []
    entail_, contra_, neutr_ = [], [], []
    for t in topics[0:52]:
        # print(t)
        # exit()
        if t=="arcompsci":
            continue
        
        journal_llm_path = './t2/'+ llm + '/data/' + t

        subdirectories = [d for d in os.listdir(journal_llm_path) if os.path.isdir(os.path.join(journal_llm_path, d))]
        journal_title_score, journal_info_score, journal_halluc_score = [], [], []

        if not os.path.exists(os.path.join("./nli_results_new", journal_llm_path)):
            os.makedirs(os.path.join("./nli_results_new", journal_llm_path))
        #保存分数
        f = open(os.path.join("./nli_results_new", journal_llm_path, t + ".txt"),'w',encoding='utf-8')
        journal_nli_scores = []

        for subd in subdirectories:
            # if subd == "Volume 13 (2020)":
            if "2023" in subd:
                json_files = find_json_files(os.path.join(journal_llm_path, subd))
                # print(os.path.join(journal_llm_path, subd))
                if not os.path.exists(os.path.join("./clean", journal_llm_path, subd)):
                        os.makedirs(os.path.join("./clean", journal_llm_path, subd))
                # json_files = find_json_files(os.path.join("./clean", journal_llm_path, subd))
                "NLI result path"
                if not os.path.exists(os.path.join("./NLI_result_new", journal_llm_path, subd)):
                        os.makedirs(os.path.join("./NLI_result_new", journal_llm_path, subd))

                # nli_scores = compute_nli_score(json_files, llm)
                # print(json_files)
                entail_scores, contra_scores, neutr_scores = compute_nli_abstract_score(json_files, llm)
            
                entail_ += entail_scores
                contra_ += contra_scores
                neutr_ += neutr_scores

                volum_score = sum(entail_scores)/len(entail_scores + contra_scores + neutr_scores)
                journal_nli_scores.append(volum_score)
                f.write("############\n")
                f.write(subd + '\n')
                f.write("volum_score:\t" + str(volum_score) +'\n')
                
                print("#########################" + t + " " + subd + "#########################")
                print("volum_score", volum_score)
        journal_nli_score = sum(journal_nli_scores)/len(journal_nli_scores)
        f.write("############\n")
        f.write(t + '\n')
        f.write("journal_nli_score:\t" + str(journal_nli_score) +'\n')
                
        print("#########################" + t  + "#########################")
        print("journal_nli_score", journal_nli_score) 
        model_nli.append(journal_nli_score)
    print(llm, sum(model_nli)/len(model_nli))
    print(len(entail_), len(contra_), len(neutr_))     # 