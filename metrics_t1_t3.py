'''
Reference evaluation for Tasks 1 and 3
The final hallucination assessment was computed using outputs from evaluate_t1.py and evaluate_t3_refer.py, including:

Dimensional accuracy metrics

Comparative analysis of first-author treatment in authorship attribution
'''
import os
from llm_api import find_json_files, read_json_file
from semantic_scholar import semantic_scholar_search
# from evaluate import save_to_csv
import csv, json
import scipy
from scipy import stats
from figure import plot_json_disciline,plot_json_disciline_all_models,plot_radar_chart
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
'''
根据evaluate.py中的比较结果，计算最后的幻觉结果
'''
def save_to_csv(data, filename, title= ['Journal Name', 'Title acc', "Overall hallu score"]):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(title)
        writer.writerows(data)

def calculate_scores_(list_score={}, list_score_pro={},first_author=False):
    # Step 1: title_correct == 1 时，list_score 中其他值为 1 的数量
    title_related_count = 0
    other_four = 0
    if float(list_score["title_correct"]) != 1.0:
        if not first_author:
            if float(list_score["author_correct"]) == 1.0:
                other_four += 1
        else:
            if float(list_score_pro["first_author"]) == 1.0:
                other_four += 1
        if float(list_score["journal_correct"]) == 1.0:
            other_four += 1
        if float(list_score["year_correct"]) == 1.0:
            other_four += 1

        if float(list_score["volume_correct"]) == 1.0:
            other_four += 1
        if float(list_score["first_page_correct"]) == 1.0:
            other_four += 1
        if float(list_score["last_page_correct"]) == 1.0:
            other_four += 1

    if other_four >= 3:
        title_related_count += 1 


    # Step 2: list_score_pro 中值为 1 时，list_score 中除去 title_correct 其他值为 1 的数量
    pro_related_count = 0
    other_four = 0
    if float(list_score_pro["title_correct_0.8"]) != 1.0 and float(list_score["title_correct"]) != 1.0:
        if not first_author:
            if float(list_score["author_correct"]) == 1.0:
                other_four += 1
        else:
            if float(list_score_pro["first_author"]) == 1.0:
                other_four += 1
        if float(list_score["journal_correct"]) == 1.0:
            other_four += 1
        if float(list_score["year_correct"]) == 1.0:
            other_four += 1
        if float(list_score["volume_correct"]) == 1.0:
            other_four += 1
        if float(list_score["first_page_correct"]) == 1.0:
            other_four += 1
        if float(list_score["last_page_correct"]) == 1.0:
            other_four += 1

    if other_four >= 3:
        pro_related_count += 1

    return title_related_count, pro_related_count

def calculate_scores(list_score={}, list_score_pro={},first_author=False):
    # Step 1: title_correct == 1 时，list_score 中其他值为 1 的数量
    '''匹配title100%'''
    title_related_count = 0
    if float(list_score["title_correct"]) == 1.0:
        if not first_author:
            if float(list_score["author_correct"]) == 1.0:
                title_related_count += 1
        else:
            if float(list_score_pro["first_author"]) == 1.0:
                title_related_count += 1
        if float(list_score["journal_correct"]) == 1.0:
            title_related_count += 1
        if float(list_score["year_correct"]) == 1.0:
            title_related_count += 1
        if float(list_score["volume_correct"]) == 1.0:
            title_related_count += 1
        if float(list_score["first_page_correct"]) == 1.0:
            title_related_count += 1
        if float(list_score["last_page_correct"]) == 1.0:
            title_related_count += 1

    pro_related_count = 0
    if float(list_score_pro["title_correct_0.8"]) == 1.0 and float(list_score["title_correct"]) != 1.0:
        if not first_author:
            if float(list_score["author_correct"]) == 1.0:
                pro_related_count += 1
        else:
            if float(list_score_pro["first_author"]) == 1.0:
                pro_related_count += 1
        if float(list_score["journal_correct"]) == 1.0:
            pro_related_count += 1
        if float(list_score["year_correct"]) == 1.0:
            pro_related_count += 1 
        if float(list_score["volume_correct"]) == 1.0:
            pro_related_count += 1
        if float(list_score["first_page_correct"]) == 1.0:
            pro_related_count += 1
        if float(list_score["last_page_correct"]) == 1.0:
            pro_related_count += 1
    if pro_related_count > 0: # 除了题目匹配之外，有另外一项也匹配
        pro_related_count = 1
    if title_related_count > 0:
        title_related_count = 1

    return title_related_count, pro_related_count


def scholar_search_references_acc(json_files):
    '''
    for each Journal search each reference from google scholar
    '''
    title_aver, info_aver, halluc = [],[],[]
    index = 0 
    find_these_paper = 0
    not_find_these_paper = 0
    for file in json_files:
        d = read_json_file(file)
        index += 1
        '''
        save path for the best candidate
        '''
        save_path = file.replace('./clean','./compare_data_fuzzy')
        if not os.path.exists('/'.join(save_path.split('/')[0:-1])):
            os.makedirs('/'.join(save_path.split('/')[0:-1]))
        
        if "Literature Review" in d.keys():
            if isinstance(d["Literature Review"], dict):
                if "References" in d["Literature Review"].keys():
                    t, r = "", d["Literature Review"]["References"]
            else:
                t, r = d["Literature Review"], d["References"]
        else:
            t = ""
            r =  d["References"]

        references_candidate = {}
        info_correct, correct_title, all = [], [], 0
        best_candidate_info = []  #用于将每个
        # print("r", r)
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

            title_results, semantic_results, flag = semantic_scholar_search(title)

            if not flag or semantic_results == {'error': 'No valid paper ids given'}:
                not_find_these_paper += 1
            else:
                find_these_paper += 1
    return find_these_paper/(find_these_paper + not_find_these_paper)

def scholar_search_references_acc_llm4(json_files):
    '''
    for each Journal search each reference from google scholar
    '''
    title_aver, info_aver, halluc = [],[],[]
    index = 0 
    find_these_paper = 0
    not_find_these_paper = 0
    for file in json_files:
        d = read_json_file(file)
        index += 1
        '''
        save path for the best candidate
        '''
        # save_path = file.replace('./clean','./compare_data_fuzzy')
        # if not os.path.exists('/'.join(save_path.split('/')[0:-1])):
        #     os.makedirs('/'.join(save_path.split('/')[0:-1]))
        
        r =  d["References"]

        references_candidate = {}
        info_correct, correct_title, all = [], [], 0
        best_candidate_info = []  #用于将每个
        # print("r", r)
        # exit()
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

            title_results, semantic_results, flag = semantic_scholar_search(title)

            if not flag or semantic_results == {'error': 'No valid paper ids given'}:
                not_find_these_paper += 1
            else:
                find_these_paper += 1
    return find_these_paper/(find_these_paper + not_find_these_paper)
def search_acc(llm="gpt-3.5"):
    llm = llm
    # llm = "meta-llama_Meta-Llama-3.1-8B-Instruct"
    # llm = "llama"
    topics = [f.name for f in os.scandir("./clean/data") if f.is_dir()]

    # topics = ["micro","neuro","nucl","nutr","orgpsych","pathmechdis","pharmtox","physchem","physiol","phyto","arplant","polisci","psych","publhealth","resource","soc","statistics","virology","vision"]
    # topics = ["micro"]

    # topics = [f.name for f in os.scandir("./llm4/gpt-3.5/data") if f.is_dir()]
    # print(len(topics))

    # 统计信息
    # directory = "llm4/meta-llama_Meta-Llama-3.1-8B-Instruct"  # 替换为你要检查的文件夹路径
    # json_files = count_json_files(directory)
    # print(f"Total JSON files: {json_files}")
    # count_data = count_json_files_in_subfolders(directory)
    # save_to_csv(count_data, './count_paper.csv')

    # exit()
    # # '''journal name'''
    # print(len(topics))
    # exit()
    model_find_acc = [] #
    for t in topics[0:52]:
        # print(t)
        # exit()
        if t=="arcompsci":
            continue
        
        journal_llm_path = './llm6/'+ llm + '/data/' + t

        subdirectories = [d for d in os.listdir(journal_llm_path) if os.path.isdir(os.path.join(journal_llm_path, d))]
        journal_title_score, journal_info_score, journal_halluc_score, journal_find_acc = [], [], [],[]

        if not os.path.exists(os.path.join("./hallu_results_fuzzy_t3", journal_llm_path)):
            os.makedirs(os.path.join("./hallu_results_fuzzy_t3", journal_llm_path))
        #保存分数
        f = open(os.path.join("./hallu_results_fuzzy_t3", journal_llm_path, t + ".txt"),'w',encoding='utf-8')
       
        for subd in subdirectories:
            # if subd == "Volume 13 (2020)":
                json_files = find_json_files(os.path.join(journal_llm_path, subd))
                # if not os.path.exists(os.path.join("./clean", journal_llm_path, subd)):
                #         os.makedirs(os.path.join("./clean", journal_llm_path, subd))
                
                '''clean data generated form LLM as json format'''
                # clean_references(json_files) #clean the generated results
                # print("finished LLM data cleaning!")

                # json_files = find_json_files(os.path.join("./clean", journal_llm_path, subd))
                json_files = find_json_files(os.path.join(journal_llm_path, subd)) #new experiments

                # title_score, info_score, halluc_score = scholar_search_references(json_files)
                find_acc = scholar_search_references_acc_llm4(json_files) #有多少文章被找到
                # f.write("############\n")
                # f.write(subd + '\n')
                # f.write("title_score:\t" + str(title_score) +'\n')
                # f.write("info_score:\t" + str(info_score) +'\n')
                # # f.write("info_score%:\t" + str(info_score/7) +'\n')
                # f.write("halluc_score:\t" + str(halluc_score) +'\n')
                # print("#########################" + t + " " + subd + "#########################")
                # print("title_score", title_score)
                # print("info_score", info_score)
                # print("info_score%", info_score/7)
                # print("halluc_score", halluc_score)
                # journal_title_score.append(title_score)
                # journal_info_score.append(info_score)
                # journal_halluc_score.append(halluc_score)
                journal_find_acc.append(find_acc)
                
        # journal_title_score_ = sum(journal_title_score)/len(journal_title_score)
        # journal_info_score_ = sum(journal_info_score)/len(journal_info_score) 
        # journal_halluc_score_ = sum(journal_halluc_score)/len(journal_halluc_score) 
        journal_find_acc_ = sum(journal_find_acc)/len(journal_find_acc)
        # print("#########################" + t + "#########################")
        # print("journal_title_score", journal_title_score_)
        # print("journal_info_score", journal_info_score_)
        # print("journal_info_score%", journal_info_score_/7)
        print("journal_find_acc_", journal_find_acc_)
        # print("journal_halluc_score", journal_halluc_score_)
        model_find_acc.append(journal_find_acc_)
        # f.write("############\n")
        # f.write(t + '\n')
        # f.write("journal_title_score:\t" + str(journal_title_score_) +'\n')
        # f.write("journal_info_score:\t" + str(journal_info_score_) +'\n')
        # f.write("journal_info_score%:\t" + str(journal_info_score_/7) +'\n')
        # f.write("journal_find_acc_%:\t" + str(journal_find_acc_) +'\n')
        # f.write("journal_halluc_score:\t" + str(journal_halluc_score_) +'\n')
    print("model_find_acc:",sum(model_find_acc)/len(model_find_acc))      

            

def plot_model_trends(data, task):
    """
    绘制每个模型在2010年之前的数据总和以及2010年至2024年的变化趋势。

    参数:
        data (dict): 每个模型包含年份和对应值的字典
    """
    # 定义年份范围
    start_year = 2000
    end_year = 2023
    
    # 初始化存储结果的字典
    model_trends = {}
    
    for model, year_data in data.items():
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
        # 转换年份为整数，处理可能的字符串 "None"
        year_data = {int(year): count for year, count in year_data.items() if year != "None"}
        
        # 计算2010年之前的总和
        pre_2000_sum = sum(value for year, value in year_data.items() if year < start_year)
        
        # 获取2010年至2024年的数据
        post_2000_trend = {year: year_data.get(year, 0) for year in range(start_year, end_year + 1)}
        
        # 保存结果
        model_trends[model] = {"pre_2000_sum": pre_2000_sum, "post_2000_trend": post_2000_trend}
    
    # 绘制折线图
    # color = {"Claude-3.5": "#925eb0", \
    #         "DeepSeek-V3": "#237B9F",\
    #         "GPT-4o":"#71BFB2",\
    #         "Qwen-2.5": "#EC817E",\
    #         "Llama-3.2":"#FEE066"}
    color = {"Claude-3.5": "#EF666E", \
            "DeepSeek-V3": "orange",\
            "GPT-4o": "#6DCA97",\
            "Qwen-2.5": "#9BABD5",\
            "Llama-3.2": "#D0CADE"}
            
    plt.figure(figsize=(16, 9))
    for model, trend_data in model_trends.items():
        years = [start_year - 1] + list(trend_data["post_2000_trend"].keys())
        values = [trend_data["pre_2000_sum"]] + list(trend_data["post_2000_trend"].values())
        
        plt.plot(years, values, color=color[model], marker="^",markersize=8, linewidth=3, label=model)
        # plt.plot(years, values, marker="o",linewidth=3, label=model)

    
    # 修改x轴标签
    x_ticks = [start_year - 1] + list(range(start_year, end_year + 1))
    x_labels = ["Pre 2000"] + [str(year) for year in range(start_year, end_year + 1)]
    # 每隔5年显示一个标签（包括 Pre 2000）
    x_ticks_display = [x for i, x in enumerate(x_ticks) if (i - 1) % 5 == 0]
    x_labels_display = [x_labels[i] for i in range(len(x_labels)) if  (i - 1) % 5 == 0]

    plt.tick_params(axis='y', labelsize=20)
    plt.xticks(x_ticks_display, x_labels_display, rotation=45, fontsize=20)

    # plt.xticks(x_ticks, x_labels, rotation=45, step=5, fontsize=20)
    
    # 添加图例和标签
    # plt.title("", fontsize=16)
    plt.xlabel("Year", fontsize=30)
    plt.ylabel("Number of True References", fontsize=30)
    plt.legend(fontsize=20, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.xticks(np.arange(1999, 2024, step=5), rotation=45)
    plt.savefig("./final_score_" + task + "/paper2year.png")

def merge_data_by_five_years(discipline, years, data):
    merged_years = [years[0]]  # Keep the first year (1999) as it is
    merged_data = [data[0]]  # Keep the first data point as it is
    # 460 articles in 500 the Biology category, 90 in Chemistry, 50 in Math- 501 ematics, 113 in Physics, 299 in Social Science, and 502 94 in Technology. 503

    discipline2totalpaper = {"Biology": 4600,"Chemistry": 900, "Mathematics": 500, "Physics": 1130, "Social Science": 2990,"Technology":940 }
    # Loop through the data from 2000 and group by 5 years
    for i in range(1, len(years) - 4, 5):
        # Sum the data for the next 5 years
        merged_years.append(years[i + 2])  # Add the year in the middle of the range (e.g., 2002 for 2000-2004)
       
        merged_data.append(sum(data[i:i + 5])/discipline2totalpaper[discipline])  # Sum the data for the next 5 years

    return merged_years, merged_data

def plot_subject_trends(task, llm, data):
    processed_data = {}
    discipline2totalpaper = {"Biology": 4600,"Chemistry": 900, "Mathematics": 500, "Physics": 1130, "Social Science": 2990,"Technology":940 }

    for subject, records in data.items():
        yearly_counts = {}
        pre_2000_total = 0
        
        for i in range(2000,2025):
            if str(i) not in records.keys():
                records[str(i)] = 0
        for year, count in records.items():
            if year == 'None':
                continue  # 跳过无效年份
            
            year = int(year)
            if year < 2000:
                pre_2000_total += count
            else:
                yearly_counts[year] = yearly_counts.get(year, 0) + count
        
        if pre_2000_total > 0:
            yearly_counts[1999] = pre_2000_total  # 归入1999年
        for key in yearly_counts.keys():
            yearly_counts[key] = yearly_counts[key]/discipline2totalpaper[subject]
        processed_data[subject] = yearly_counts
    
    plt.figure(figsize=(12, 6))
    
    for subject, yearly_counts in processed_data.items():
        years = sorted(yearly_counts.keys())
        counts = [yearly_counts[year] for year in years]
        
        print(llm,subject)
        print(merge_data_by_five_years(subject, years, counts))
        plt.plot(years, counts, marker='o', label=subject)
    
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Annual Count Trends by Subject')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(np.arange(1999, 2024, step=5), rotation=45)
    plt.savefig("./final_score_" + task + "/" + llm + "paper2year.png")

if __name__ == "__main__":
    """
    读取学科分类
    """
    tasks = ["t1"]
    for task in tasks:
        print("++++++++++++++++++++++++++++++",task, "++++++++++++++++++++++")
        llms = ["claude"] #,"deepseek","Qwen_Qwen2.5-72B-Instruct","claude","gpt-4o-new","meta-llama_Llama-3.2-3B-Instruct"]
        # llms = ["deepseek"]
        first_authors=[True] 
        for first_author in first_authors:
            paper2year = {} #在任务1中评价生成的文章中没有幻觉的那些都是哪一年的
            paper_2descipline_2year, paper_2papertype_2descipline, paper_2openAccess_2descipline, paper_2citation_2descipline = {}, {}, {}, {} ##在任务1中评价生成的文章中没有幻觉的那些都是哪一年的,且在不同学科上是否会有差异
            for llm in llms:
                paper2year[llm] = {}
                paper_2descipline_2year[llm] = {}
                paper_2openAccess_2descipline[llm] = {} #"deepseek":{"social acience":{"True":200, "False":}}
                paper_2papertype_2descipline[llm] = {} #"deepseek":{"social acience":{"review":200, "journalarticle":222,"Null":200}}
                # paper_2papertype_2descipline[llm] = {} #"deepseek":{"social acience":{"review":200, "journalarticle":222,"Null":200}}
                paper_2citation_2descipline[llm] = {}
                with open('./discipline.json', 'r', encoding='utf-8') as f:
                    discipline = json.load(f)
                for key in discipline.keys():
                    paper_2descipline_2year[llm][key] = {} #{gtp-4o:{math:{2020:1, 2021:2}}} 
                    paper_2openAccess_2descipline[llm][key] = {"False":0, "True":0}
                    paper_2citation_2descipline[llm][key] = {"citationCount": 0, "count": 0}
                    paper_2papertype_2descipline[llm][key] = {"review":0, "journal":0, "null":0}

                discipline2acc = {}
                topics = [f.name for f in os.scandir( './compare_data_fuzzy_' + task + '/'+ llm  +'/data/') if f.is_dir()]

                all_correct = {"title_correct":0, "author_correct":0, "journal_correct":0, "year_correct":0,"volume_correct":0, "first_page_correct":0, "last_page_correct":0} 
                all_count = {"title_correct":0, "author_correct":0, "journal_correct":0, "year_correct":0,"volume_correct":0, "first_page_correct":0, "last_page_correct":0} 
                all_correct_no_hallu = {"first_author":0, "title_correct":0, "author_correct":0, "journal_correct":0, "year_correct":0,"volume_correct":0, "first_page_correct":0, "last_page_correct":0} 

                sum_, title_acc_, title_fuzzy_, title_acc_other, title_fuzzy_other = 0, 0, 0, 0, 0
                search = 0
                title_80 = []
                if not os.path.exists('./final_score_' + task + '/'+ llm):
                    os.makedirs('./final_score_' + task + '/'+ llm)
                save_path = os.path.join('./final_score_' + task + '/', llm, 'hallu_score.csv')
                true_list = []
                no_hallu_paper_count = 0
                title_fuzzy_acc, halluc_score_model = [], []
                journal_hallu_list = []
                for t in topics:
                    journal_llm_path = './compare_data_fuzzy_' + task + '/'+ llm  + '/data/' + t
                    sum_r_journal, refer_right_journal, title_right_journal = 0, 0, 0
                    subdirectories = [d for d in os.listdir(journal_llm_path) if os.path.isdir(os.path.join(journal_llm_path, d))]
                    for subd in subdirectories:
                        #2023年这一期
                        json_files = find_json_files(os.path.join(journal_llm_path, subd))
                        title_correct_journal, halluc_score_journal = [], []
                        for f in json_files:
                            #某一篇文章对应的幻觉
                            # print(f)
                            
                            sum_r_paper, refer_right_paper, title_right_paper = 0, 0, 0
                            text = read_json_file(f)
                            article_score, artcle_all_score = 0.0, 0.0
                            if task == "t3" and first_author:
                                abstract_path = f.replace("compare_data_fuzzy_" + task, "t3_true_refer_abstract")#存放哪些真实文献的abstract

                                # if os.path.exists(abstract_path):
                                #     continue
                                if not os.path.exists("/".join(abstract_path.split("/")[0:-1])):
                                    os.makedirs("/".join(abstract_path.split("/")[0:-1]))
                                abstract = []
                            # print(len())
                            for z in text: 
                                # print(z)
                                # exit()
                                sum_ += 1  #用于计算所有生成的文献
                                sum_r_journal += 1 #用于计算一个期刊中生成的所有文献
                                sum_r_paper += 1 
                                list_score = z["score_list"]
                                list_score_pro = z["c_score_pro"]
                                title_right_journal += list_score["title_correct"] #title正确，下面还要加上哪些title80%正确的情况
                                title_right_paper += list_score["title_correct"] #title正确，下面还要加上哪些title80%正确的情况
                                for key in list_score.keys():
                                    if list_score[key] != -1:
                                        all_correct[key] += list_score[key]
                                        all_count[key] += 1
                                        if key == "title_correct" and list_score[key] == 0:
                                            # all_correct[key] += list_score_pro["title_correct_0.8"]
                                            title_right_journal += list_score_pro["title_correct_0.8"]
                                            title_right_paper += list_score_pro["title_correct_0.8"]

                                        if key == "title_correct" or key == "author_correct" or key == "journal_correct":
                                            if key == "title_correct":
                                                article_score += float(list_score_pro["title_correct_0.8"]) * 2.0
                                                artcle_all_score += 2.0
                                            else:
                                                article_score += float(list_score[key]) * 2.0
                                                artcle_all_score += 2.0
                                        else:
                                            article_score += float(list_score[key]) * 1.0
                                            artcle_all_score += 1.0
                                # print(artcle_all_score)
                                # print(article_score)
                                # exit()
                                if article_score/artcle_all_score >= 0.5:
                                    true_list.append(1)
                                # true_list.append(article_score/artcle_all_score)

                                list_score_pro = z["c_score_pro"]
                                # if list_score_pro["title_correct_0.8"] == 1 and list_score["title_correct"] != 1 and list_score["journal_correct"] == 1:
                                # print(z)
                                search += z["search_score"]
                                title_related_count, pro_related_count = calculate_scores(list_score=list_score, list_score_pro=list_score_pro, first_author=first_author)
                                if first_author:
                                    if title_related_count == 1 or pro_related_count == 1:
                                        
                                        for key in discipline.keys():
                                            if t in discipline[key]:
                                                if z['year'] in paper_2descipline_2year[llm][key].keys():
                                                    paper_2descipline_2year[llm][key][z['year']] += 1
                                                else:
                                                    paper_2descipline_2year[llm][key][z['year']] = 1

                                                if "publicationTypes" in z.keys() and z["publicationTypes"]:
                                                    if "Review" in z["publicationTypes"]:
                                                        # paper_2openAccess_2descipline[llm][key] = {}
                                                        # paper_2citation_2descipline[llm][key] = {}
                                                        paper_2papertype_2descipline[llm][key]["review"] += 1
                                                    elif "JournalArticle" in z["publicationTypes"]:
                                                        paper_2papertype_2descipline[llm][key]["journal"] += 1
                                                    else:
                                                        paper_2papertype_2descipline[llm][key]["null"] += 1
                                                # print(z)
                                                # exit()
                                                if "citationCount" in z.keys() and z["citationCount"]:
                                                    paper_2citation_2descipline[llm][key]["count"] += 1
                                                    paper_2citation_2descipline[llm][key]["citationCount"] += z["citationCount"]
                                                    # if key == "Mathematics":
                                                    #     print(key, z["citationCount"])

                                                if "isOpenAccess" in z.keys() and z["isOpenAccess"]:
                                                    paper_2openAccess_2descipline[llm][key]["True"] += 1
                                                else:
                                                    paper_2openAccess_2descipline[llm][key]["False"] += 1
                                        if z['year'] in paper2year[llm].keys():
                                            paper2year[llm][z['year']] += 1
                                        else:
                                            paper2year[llm][z['year']] = 1
                                        if task == "t3":
                                        # if title_related_count_ == 1 or pro_related_count_ == 1:
                                            abstract.append(z)
                                title_acc_ += title_related_count #title 完全正确，且有其他一项正确
                                refer_right_journal += title_related_count
                                refer_right_paper += title_related_count

                                title_fuzzy_ += pro_related_count #title 80%完全正确，且有其他一项正确
                                refer_right_journal += pro_related_count
                                refer_right_paper += pro_related_count
                                #title 不对，但其他有三项对
                                title_related_count_, pro_related_count_ = calculate_scores_(list_score=list_score, list_score_pro=list_score_pro, first_author=first_author)
                                if first_author:
                                    if title_related_count_ == 1 or pro_related_count_ == 1:
                                        for key in discipline.keys():
                                            if t in discipline[key]:
                                                if z['year'] in paper_2descipline_2year[llm][key].keys():
                                                    paper_2descipline_2year[llm][key][z['year']] += 1
                                                else:
                                                    paper_2descipline_2year[llm][key][z['year']] = 1
                                                if "publicationTypes" in z.keys() and z["publicationTypes"]:
                                                    if "Review" in z["publicationTypes"]:
                                                        # paper_2openAccess_2descipline[llm][key] = {}
                                                        # paper_2citation_2descipline[llm][key] = {}
                                                        paper_2papertype_2descipline[llm][key]["review"] += 1
                                                    elif "JournalArticle" in z["publicationTypes"]:
                                                        paper_2papertype_2descipline[llm][key]["journal"] += 1
                                                    else:
                                                        paper_2papertype_2descipline[llm][key]["null"] += 1
                                                if "citationCount" in z.keys() and z["citationCount"]:
                                                    paper_2citation_2descipline[llm][key]["count"] += 1
                                                    paper_2citation_2descipline[llm][key]["citationCount"] += z["citationCount"]
                                                    # if key == "Mathematics":
                                                    #     print(key, z["citationCount"])
                                                if "isOpenAccess" in z.keys() and z["isOpenAccess"]:
                                                    paper_2openAccess_2descipline[llm][key]["True"] += 1
                                                else:
                                                    paper_2openAccess_2descipline[llm][key]["False"] += 1
                                        
                                        if z['year'] in paper2year[llm].keys():
                                            paper2year[llm][z['year']] += 1
                                        else:
                                            paper2year[llm][z['year']] = 1
                                        if task == "t3":
                                        # if title_related_count_ == 1 or pro_related_count_ == 1:
                                            abstract.append(z)
                                title_acc_other += title_related_count_ #title 不正确正确，且有其他3项以上正确
                                title_fuzzy_other += pro_related_count_ #title 80%也不正确，且有其他3项以上正确
                                
                                refer_right_journal += title_related_count_
                                refer_right_journal += pro_related_count_
                                refer_right_paper += title_related_count_
                                refer_right_paper += pro_related_count_
                                if title_related_count == 1 or pro_related_count == 1 or title_related_count_== 1 or pro_related_count_ == 1:
                                    #说明当前paper不是幻觉,将当前文章的各个元素进行统计
                                    for key in list_score.keys():
                                        if list_score[key] != -1:
                                            # if  key != all_correct_no_hallu["title_correct"]
                                            all_correct_no_hallu[key] += list_score[key]
                                    if list_score_pro["first_author"] ==1:
                                        all_correct_no_hallu["first_author"] += list_score_pro["first_author"]
                                    if  list_score["title_correct"] != 1 and list_score_pro["title_correct_0.8"] ==1:
                                        all_correct_no_hallu["title_correct"] += list_score_pro["title_correct_0.8"]
                            if task == "t3" and first_author:
                                # print(len(abstract), refer_right_paper)
                                # exit()
                                with open(abstract_path, 'w', encoding='utf-8') as wa:
                                    json.dump(abstract, wa, ensure_ascii=False)

                            title_fuzzy_acc.append(title_right_paper/sum_r_paper) #100%-80%的标题正确的准确率，
                            acc_paper = (refer_right_paper/sum_r_paper) #每篇文章的幻觉
                            title_correct_journal.append(title_right_paper/sum_r_paper)
                            halluc_score_journal.append(acc_paper)  #幻觉
                            halluc_score_model.append(acc_paper)
                            for key in discipline.keys():
                                if t in discipline[key]:
                                    if key not in discipline2acc.keys():
                                        discipline2acc[key] = [acc_paper]
                                    else:
                                        discipline2acc[key].append(acc_paper)
                        journal_hallu_list.append((t, sum(title_correct_journal)/len(title_correct_journal) ,(sum(halluc_score_journal)/len(halluc_score_journal))))
                save_to_csv(journal_hallu_list, save_path)
               
                print("##########################", llm, "first_author", first_author,"#########################")
                if first_author:
                    save_path_1 = os.path.join('./final_score_' + task + '/', llm, '其他统计结果（first author）.txt')
                else:
                    save_path_1 = os.path.join('./final_score_' + task + '/', llm, '其他统计结果.txt')
                with open(save_path_1, 'w', encoding='utf-8') as f:
                    print("检索率:",search/sum_)
                    print("生成的文献总数：", sum_)    
                    f.write("生成的文献总数：\t" + str(sum_) + '\n')
                    print("生成的文献总数：", sum_), 
                    print("title 80-100acc", sum(title_fuzzy_acc)/len(title_fuzzy_acc))
                    f.write("title 80-100acc" + str(sum(title_fuzzy_acc)/len(title_fuzzy_acc)) +'\n')
                    # if task == 't3':
                    print("average acc", sum(halluc_score_model)/len(halluc_score_model))
                    f.write("average acc" +str(sum(halluc_score_model)/len(halluc_score_model)) + '\n')
                    # elif task =='t1':
                    #     print("average halluc", sum(halluc_score_model)/len(halluc_score_model))
                    #     f.write("average hallu" + str(1 - sum(halluc_score_model)/len(halluc_score_model)) + '\n')

                    # print("分子每个类别中正确的项，分母是所有生成的该项")
                    # f.write("分子每个类别中正确的项，分母是所有生成的该项\n")

                    # for key in all_correct.keys():
                    #     print(key, all_correct[key]/all_count[key])
                    #     f.write(key + '\t' + str(all_correct[key]/all_count[key]) + '\n' )
                    #  print("分子每个类别中正确的项，分母是所有生成的该项")
                    print("分子是no hallucination paper中每个类别中正确的项，分母是所有生成的该项")
                    f.write("分子是no hallucination paper中每个类别中正确的项，分母是所有生成的该项\n")

                    for key in all_correct_no_hallu.keys():
                        if key != "first_author":
                            print(key, all_correct_no_hallu[key]/all_count[key])
                            f.write(key + '\t' + str(all_correct_no_hallu[key]/all_count[key]) + '\n' )
                        else:
                            print(key, all_correct_no_hallu[key]/all_count["author_correct"])
                            f.write(key + '\t' + str(all_correct_no_hallu[key]/all_count["author_correct"]) + '\n' )
                
                if first_author:
                    '''进行t-test'''
                    biology_data = discipline2acc["Biology"]
                    chemistry_data = discipline2acc["Chemistry"]
                    math_data = discipline2acc["Mathematics"]
                    physics_data = discipline2acc["Physics"]
                    sociology_data = discipline2acc["Social Science"]
                    f_stat, p_val = stats.f_oneway(biology_data, chemistry_data, math_data, physics_data, sociology_data)
                    
                    print(f"ANOVA: F-statistic = {f_stat}, P-value = {p_val}")
                    anova_path = os.path.join('./final_score_' + task + '/', llm, 'anova.txt')
                    with open(anova_path, 'w', encoding='utf-8') as w:
                        w.write(f"ANOVA: F-statistic = {f_stat}, P-value = {p_val}")


                    # t_stat, p_val = stats.ttest_ind(biology_data, chemistry_data)
                    # print(f"Biology vs Chemistry: T-statistic = {t_stat}, P-value = {p_val}")

                    # 比较 Biology 和 Math
                    # t_stat, p_val = stats.ttest_ind(biology_data, math_data)
                    # print(f"Biology vs Math: T-statistic = {t_stat}, P-value = {p_val}")

                    # # 比较 Biology 和 Physics
                    # t_stat, p_val = stats.ttest_ind(biology_data, physics_data)
                    # print(f"Biology vs Physics: T-statistic = {t_stat}, P-value = {p_val}")

                    # # 比较 Biology 和 Sociology
                    # t_stat, p_val = stats.ttest_ind(biology_data, sociology_data)
                    # print(f"Biology vs Sociology: T-statistic = {t_stat}, P-value = {p_val}")

                    # exit()
            if first_author:
                print("开始画图")
                plot_json_disciline_all_models('./final_score_' + task + '/')
        if first_author:
            # print(paper2year)
            for key in paper_2descipline_2year.keys():
                
                plot_subject_trends("t1", key, paper_2descipline_2year[key])
                                        # exit()
            plot_model_trends(paper2year,task)
        print(paper_2papertype_2descipline)
        with open(os.path.join('./final_score_' + task + '/', llm, 'paperType.json'),'w',encoding='utf-8') as w1:
            json.dump(paper_2papertype_2descipline, w1, ensure_ascii=False)
        
        print(paper_2openAccess_2descipline)
        with open(os.path.join('./final_score_' + task + '/', llm, 'AccessType.json'),'w',encoding='utf-8') as w2:
            json.dump(paper_2openAccess_2descipline, w2, ensure_ascii=False)
        for key in paper_2citation_2descipline.keys():
            for k in paper_2citation_2descipline[key].keys():
                if paper_2citation_2descipline[key][k]["citationCount"] == 0 or paper_2citation_2descipline[key][k]["count"] == 0:
                    paper_2citation_2descipline[key][k]["Average"]=0
                else:
                    paper_2citation_2descipline[key][k]["Average"] = paper_2citation_2descipline[key][k]["citationCount"]/paper_2citation_2descipline[key][k]["count"]
        print(paper_2citation_2descipline)
        with open(os.path.join('./final_score_' + task + '/', llm, 'citationAverage.json'),'w',encoding='utf-8') as w3:
            json.dump(paper_2citation_2descipline, w3, ensure_ascii=False)
       


            # plot_radar_chart('./final_score_t1/')



        # {'title_correct': 28770, 'author_correct': 1831, 'journal_correct': 13583, 'year_correct': 8034, 'volume_correct': 4291, 'first_page_correct': 5249, 'last_page_correct': 3701}# 
        # 14631 2676 91291 0.16026771532790746 0.029312856689049305`