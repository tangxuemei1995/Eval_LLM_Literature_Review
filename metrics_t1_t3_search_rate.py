'''
The retrieval rate of titles in the reference documents for Task 1 and Task 3
'''
import os
from llm_api import find_json_files, read_json_file
from semantic_scholar import semantic_scholar_search
# from evaluate import save_to_csv
import csv, json
import scipy
from scipy import stats
from figure import plot_json_disciline,plot_json_disciline_all_models,plot_radar_chart

def save_to_csv(data, filename, title= ['Journal Name', 'Title acc', "Overall hallu score"]):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(title)
        writer.writerows(data)



def references_rate(json_files, task):
    '''
    for each Journal search each reference from semantic scholar
    '''
    volume_title_search_rate = []
    
    for file in json_files:
        d = read_json_file(file)
        r = d["References"]

        search_title, generated_title = 0, 0 #可以检索到的title，以及生成的title
        if isinstance(r,dict):
            r_ = r
            r = []
            for key in r_.keys():
                r.append(r_[key])
        for paper in r:
            paper_ = paper
            paper = {}
            for key in paper_.keys():
                paper[key.lower()] = paper_[key]
            if "title" not in paper.keys():
                continue
            title = paper["title"]
            generated_title += 1
            title_results, semantic_results, flag = semantic_scholar_search(title)
            if flag and semantic_results != {'error': 'No valid paper ids given'}:
                search_title += 1
        volume_title_search_rate.append(search_title/generated_title)
    # print(volume_title_search_rate)
    # exit()
    return volume_title_search_rate

if __name__ == "__main__":
    tasks = ["t1","t3"]
    for task in tasks:
        print("++++++++++++++++++++++++++++++",task, "++++++++++++++++++++++")
        llms = ["deepseek"]  #,""Qwen_Qwen2.5-72B-Instruct","claude","gpt-4o-new","meta-llama_Llama-3.2-3B-Instruct"]
        for llm in llms:
            model_title_seacrh_rate = []
            with open('./discipline.json', 'r', encoding='utf-8') as f:
                discipline = json.load(f)
            discipline2acc = {}
            topics = [f.name for f in os.scandir( './clean/data/') if f.is_dir()]
            for t in topics:
                if t in ["arcompsci"]:
                    continue
                journal_llm_path = os.path.join("./", task, llm, "data", t)
                subdirectories = [d for d in os.listdir(journal_llm_path) if os.path.isdir(os.path.join(journal_llm_path, d))]
                volume_path = os.path.join(journal_llm_path, subdirectories[0])
                json_files = find_json_files(volume_path)
                volume_title_search_rate = references_rate(json_files, task)
                model_title_seacrh_rate += volume_title_search_rate

            print("####################", task, llm, "####################")
            print(len(model_title_seacrh_rate))
            print("title search rate: ", sum(model_title_seacrh_rate)/len(model_title_seacrh_rate))
            with open(os.path.join("./final_score_" + task, llm, "title_search_rate.txt"), 'w', encoding='utf-8') as w:
                w.write("title search rate: " + str(sum(model_title_seacrh_rate)/len(model_title_seacrh_rate)))
# /Users/tangtang/Desktop/academic-review/final_score_t1/meta-llama_Llama-3.2-3B-Instruct
                
                
                