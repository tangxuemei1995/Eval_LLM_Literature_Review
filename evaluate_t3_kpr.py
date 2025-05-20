'''
Evaluate the literature review section in Task 3. 
The original article generated key points using GPT-4.
This file computes the entailment relationship between key points using GPT-4.
Results are stored in ./final_score_t3.
'''
from google_api import find_google_scholar
from llm_api import find_json_files, read_json_file

import re
import time
import csv
from scipy import stats
from bing_api import query_bing_return_mkt
from google_search_api import google_search
from semantic_scholar import semantic_scholar_search
from clean_llm_data import clean_references_for_second
# from evaluate import gpt
# import litellm
import openai, requests
import matplotlib.pyplot as plt
import numpy as np
import os,json
import pandas as pd
from matplotlib import rc

openai.api_key = "" 



def huggingface_api(system_prompt, user_prompt, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    
    API_URL = "https://api-inference.huggingface.co/models/" + model_id
    headers = {"Authorization": f"Bearer your key"}
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
        model="gpt-4o-2024-08-06", 
        messages=[{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=3700)  
    result = completion["choices"][0][ "message"]["content"]
    return result

# f = open('./prompt3-gpt.txt', 'r', encoding='utf-8')
# # f.read()
# system_prompt = f.read().strip()

# f = open('./prompts/prompt_nli.txt', 'r', encoding='utf-8')
# # f.read()
f = open('./prompts/point_nli.txt', 'r', encoding='utf-8')

system_prompt_nli = f.read().strip()


             

def compute_keypoint_nli_score(json_files, model_id):
    
    volume_entail_scores = []
    index = 0 
    for file in json_files:
        entail_scores = []
        contra_scores = []
        neutr_scores = []
        # print("#########################################" + file + "##############################################" )
        index += 1
        save_path_nli =os.path.join("point_NLI_result_new", file)
        results = {}
        
        if os.path.exists(save_path_nli):
            with open(save_path_nli, 'r', encoding='utf-8') as f:
                results = json.load(f)
        if results == {}:
            '''
            save path for the best candidate
            '''
            generated = read_json_file(file)
            if "Literature Review" in generated.keys():
                generated_liter = generated["Literature Review"]
            else:
                generated_liter = ""
                print("no literature in this file,", file)
                continue
            point_file = file.replace("./t3/" + model_id, "./point/gpt-4o")
            points = read_json_file(point_file)
            claims = {}
            for key in points.keys():
                new_key = key.replace("Point", "Claim")
                claims[new_key] = points[key]
            # print("claims", claims)
            # print("generated literature", generated_liter)
            # exit()

            # print(len(system_prompt_nli) + len(abstract1) +len(t))
            response = gpt_4o(system_prompt_nli, "Claims:\n" + str(claims) + '\n' + "Document:\n" + generated_liter)
            response = response.lower()
            response = json.loads(response)
            # print(response)

            results = {"Claims": claims , "Document": generated_liter, "Response":response}
            with open(save_path_nli,'w',encoding='utf-8') as f:
                  json.dump(results, f, ensure_ascii=False)
        res = results["Response"]
        for key in res.keys():
            if res[key] == "yes":
                entail_scores.append(1)
            elif res[key] == "no":
                contra_scores.append(1)
            elif res[key] == "neutral":
                neutr_scores.append(1)
            elif res[key] == "supports":
                entail_scores.append(1)
        volume_entail_scores.append(sum(entail_scores)/(sum(entail_scores) + sum(contra_scores) + sum(neutr_scores)))
    # print(volume_entail_scores)  
    # exit()
    return volume_entail_scores
    


def figure_t3_kpr(model2dis):
    # 数据
    # rc('font', family='Times New Roman')


    # 计算正确和错误的占比
    # fields = ['Biology', 'Chemistry','Mathematics','Physics','Social Science', 'Technology']
    data = []
    for model in model2dis:
        discipline2scores = model2dis[model] 

        for dis in discipline2scores.keys():
            data.append((dis, model, sum(discipline2scores[dis])/len(discipline2scores[dis])*100))
    df = pd.DataFrame(data, columns=['Despline', 'Model', 'Score'])
    pivot_df = df.pivot(index='Despline', columns='Model', values='Score')
    journal_names = pivot_df.index
    models = pivot_df.columns
    scores = pivot_df.values

    x = np.arange(len(journal_names))
    width = 0.14  # 每个柱的宽度
    gap = 0.008   # 组内柱之间的间距

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(16, 9))

    # 绘制每个模型的数据柱
    df_h = {}
    for i, model in enumerate(models):
        if model == "meta-llama_Llama-3.2-3B-Instruct":
            model = "Llama-3.2"
        if model.startswith("Qwen"):
            model = "Qwen-2.5"
        if model.startswith("claude"):
            model = "Claude-3.5"
        if model.startswith("gpt"):
            model = "GPT-4o"
        if model.startswith("deep"):
            model = "DeepSeek-V3"
        df_h[model] = scores[:, i]
    # print(df_h)
    # exit()
    models = ['Claude-3.5', 'DeepSeek-V3','GPT-4o', 'Qwen-2.5', 'Llama-3.2']
    # color = ['#E47178',"#AD0B08", '#75C298', '#A0ADD0', '#F5DC75']
    color = ['#EF666E', 'orange', '#6DCA97', '#9BABD5', '#D0CADE']
    for i in range(len(models)):
        adjusted_x = x + i * (width + gap)  # 添加间距调整
        ax.bar(adjusted_x, df_h[models[i]], width, label=models[i], color=color[i],edgecolor='gray', linewidth=1.0)

    # 添加标签和标题
    # ax.set_xlabel('Discipline', fontsize=25)
    ax.set_ylabel('KPR', fontsize=30)
    # ax.set_title('Model Performance Across Different Discipline Categories', fontsize=)
    ax.set_xticks(x + (len(models) - 1) * (width + gap) / 2)  # 调整x轴刻度位置
    ax.set_xticklabels(journal_names, rotation=30, ha='center', fontsize=25)
    ax.tick_params(axis='y', labelsize=25)  # 调整Y轴刻度字体大小

    ax.legend(loc='upper center',fontsize=20, bbox_to_anchor=(0.5,1.2), ncol=5)#bbox_to_anchor=(0.5, -0.2)
    
    # 调整整体布局和字体大小
    plt.tight_layout()
    plt.savefig("./final_score_t3/"+"kpr_score.png", dpi=300)


    # 显示图例
    # plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.legend(loc='upper center',fontsize=25, bbox_to_anchor=(0.5, -0.1),ncol=4)

    # 显示图表
    plt.ylim((40,95))
    plt.tight_layout()

    # 保存图表
    plt.savefig('./figures/t2_true_entailment.png')
    # plt.show()

if __name__ == "__main__":
    model2discipline = {}
    llms = ["deepseek","gpt-4o-new", "claude", "Qwen_Qwen2.5-72B-Instruct", "meta-llama_Llama-3.2-3B-Instruct"]
    for llm in llms:
        with open('./discipline.json', 'r', encoding='utf-8') as f:
            discipline = json.load(f)
        discipline2acc = {}
        topics = [f.name for f in os.scandir("./clean/data") if f.is_dir()]
        model_nli = []
        entail_, contra_, neutr_ = [], [], []
        for t in topics[0:52]:
            if t == "arcompsci":
                continue
            journal_llm_path = './t3/' + llm + '/data/' + t

            subdirectories = [d for d in os.listdir(journal_llm_path) if os.path.isdir(os.path.join(journal_llm_path, d))]
            journal_title_score, journal_info_score, journal_halluc_score = [], [], []

            if not os.path.exists(os.path.join("./point_nli_results_new", journal_llm_path)):
                os.makedirs(os.path.join("./point_nli_results_new", journal_llm_path))
            #保存分数
            f = open(os.path.join("./point_nli_results_new", journal_llm_path, t + ".txt"),'w',encoding='utf-8')
            journal_nli_scores = []

            for subd in subdirectories:
                if "2023" in subd:
                    json_files = find_json_files(os.path.join(journal_llm_path, subd))
                    # print(os.path.join(journal_llm_path, subd))
                    if not os.path.exists(os.path.join("./clean", journal_llm_path, subd)):
                            os.makedirs(os.path.join("./clean", journal_llm_path, subd))
                    # json_files = find_json_files(os.path.join("./clean", journal_llm_path, subd))
                    "NLI result path"
                    if not os.path.exists(os.path.join("./point_NLI_result_new", journal_llm_path, subd)):
                            os.makedirs(os.path.join("./point_NLI_result_new", journal_llm_path, subd))

                    volume_entail_scores = compute_keypoint_nli_score(json_files, llm)
                    journal_nli_score = sum(volume_entail_scores)/len(volume_entail_scores)
                    # f.write("############\n")
                    # f.write(subd + '\n')
                    # f.write("volum_score:\t" + str(journal_nli_score) +'\n')    
                    # print("#########################" + t + " " + subd + "#########################")
                    # print("journal_score", journal_nli_score)
                    # journal_nli_score = sum(journal_nli_scores)/len(journal_nli_scores)
                    f.write("############\n")
                    f.write(t + '\n')
                    f.write("journal_nli_score:\t" + str(journal_nli_score) +'\n')
                    model_nli += volume_entail_scores
            for key in discipline.keys():
                if t in discipline[key]:
                    if key not in discipline2acc.keys():
                        discipline2acc[key] = volume_entail_scores
                    else:
                        discipline2acc[key] += volume_entail_scores
            # print(discipline2acc) 
        biology_data = discipline2acc["Biology"]
        chemistry_data = discipline2acc["Chemistry"]
        math_data = discipline2acc["Mathematics"]
        physics_data = discipline2acc["Physics"]
        sociology_data = discipline2acc["Social Science"]
        print("#####################", llm, "#######################")
        print("sociology_data",len(sociology_data))
        print("math_data",len(math_data))
        print("physics_data",len(physics_data))
        print("chemistry_dat", len(chemistry_data))
        print("biology_data", len(biology_data))

        f_stat, p_val = stats.f_oneway(biology_data, chemistry_data, math_data, physics_data, sociology_data)
        print(llm, f"ANOVA: F-statistic = {f_stat}, P-value = {p_val}")
        anova_path = os.path.join('./final_score_t3' + '/', llm, 'liter_anova.txt')
        with open(anova_path, 'w', encoding='utf-8') as w:
            w.write(f"ANOVA: F-statistic = {f_stat}, P-value = {p_val}")
        # print(len(model_nli))
        # exit()
        print(llm,"KPR", sum(model_nli)/len(model_nli))
        liter_path = os.path.join('./final_score_t3' + '/', llm, 'liter_KPR.txt')
        
        with open(liter_path, 'w', encoding='utf-8') as w:
            w.write(f"KPR = {sum(model_nli)/len(model_nli)}")
        model2discipline[llm] = discipline2acc
    figure_t3_kpr(model2discipline)
        
        # print(len(entail_), len(contra_), len(neutr_))     # 