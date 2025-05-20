'''evaluate_t1_t3.py中调用'''
import matplotlib.pyplot as plt
import pandas as pd
import csv, os
import numpy as np
from matplotlib import rc
import json


with open('./discipline.json', 'r', encoding='utf-8') as f:
        discipline = json.load(f)

# discipline = {
#     "Natural Sciences and Mathematics": [
#     "cellbio", "micro", "nucl", "arplant", "genom", "biochem", "ento", "marine", "bioeng", "energy", "anchem",
#     "conmatphys", "vision", "fluid", "pathmechdis", "biophys", "earth", "physchem", "neuro",
#     "cancerbio", "genet", "virology", "astro", "ecolsys", "physiol",
#   ],
#   "Mathematics":["matsci", "statistics",  "biodatasci"],
#   "Social Sciences": [
#    "devpsych", "clinpsy", "psych", "orgpsych","soc", "financial", "publhealth", "criminol", "resource", "lawsocsci", "economics", "polisci",   "linguistics"
#   ],
# #   "Philosophy and Psychology": [
# #     "devpsych", "clinpsy", "psych", "orgpsych"
# #   ],
#   "Technology": [
#     "chembioeng", "nutr", "bioeng", "energy", "food", "control", "animal", "phyto","med"
#   ]
# }

# discipline = {"Biomedical&Life Sciences":["anchem","animal","biochem","biodatasci","bioeng","biophys","cancerbio",\
#                             "cellbio","chembioeng","clinpsy","devpsych","ecolsys","ento","food",\
#                             "genet","genom","immunol","marine","med","micro","neuro","nutr",\
#                             "pathmechdis","pharmtox","physiol","phyto","arplant","psych",
#                             "publhealth","statistics","virology","vision"],

# "Physical Sciences":["anchem","astro","bioeng","biophys","chembioeng","arcompsci"\
#                      "conmatphys","control","earth","energy","fluid",\
#                         "matsci","marine","nucl","physchem","statistics"],

# "Social Sciences":["psych","publhealth", "criminol",\
#                     "soc","statistics","resource","polisci", \
#                     "orgpsych",  "linguistics","lawsocsci",\
#                     "financial","astro","clinpsy","devpsych",\
#                     "economics","energy"],
# "Economics":["financial","economics","resource","arcompsci"]
# }


def plot_json_counts(csv_filename):
    # 读取 CSV 文件
    df = pd.read_csv(csv_filename)
    
    # 绘制柱形图
    plt.figure(figsize=(15, 6))

    plt.bar(df['Folder Name'], df['JSON File Count'], color='skyblue')
    plt.xlabel('Journal Name')
    plt.ylabel('Number of papers')
    plt.title('Number of papers in Each Journal')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存图形为文件
    plt.savefig('./json_file_counts.png')
    # plt.show()
    print(df['Folder Name'])



def save_to_csv(data, filename, title="acc"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Discipline Name', title])
        writer.writerows(data)

def plot_nli_disciline(csv_filename, lim_overall=(90,98), title='External'):
    # 读取 CSV 文件
    df = pd.read_csv(csv_filename)
    new_halluc_score, new_title_score = {}, {}
    for key in discipline.keys():
        new_halluc_score[key] = []
        new_title_score[key] = []
    for i in range(len(df['Discipline Name'])):
        for key in discipline.keys():
            if df['Discipline Name'][i] in discipline[key]:
                new_halluc_score[key].append(float(df['NLI score'][i]))
                # new_title_score[key].append(float(df['Title acc'][i]))
    
    '''hulla score'''
    plt.figure(figsize=(10, 6))
    disc, count = [],[]
    disc_count = []
    for key in new_halluc_score.keys():
        disc.append(key)
        count.append(sum(new_halluc_score[key])/len(new_halluc_score[key])*100)
        disc_count.append((key, sum(new_halluc_score[key])/len(new_halluc_score[key])*100))

    save_to_csv(disc_count, csv_filename.replace("nli_score", "dicspline_nli_score"))

    plt.bar(disc, count, color='skyblue')
    plt.xlabel('Discipline')
    plt.ylabel(title + ' NLI Score')
    plt.title(title +' NLI Score of Each Discipline')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.ylim(lim_overall)

    # 保存图形为文件

    plt.savefig(csv_filename.replace('.csv',"_disciline_halllucination_score.png"), title='hallu score')
    

def plot_json_counts(csv_filename):
    # 读取 CSV 文件
    df = pd.read_csv(csv_filename)
    
    # 绘制柱形图
    plt.figure(figsize=(15, 6))

    plt.bar(df['Folder Name'], df['JSON File Count'], color='skyblue')
    plt.xlabel('Journal Name')
    plt.ylabel('Number of papers')
    plt.title('Number of papers in Each Journal')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存图形为文件
    plt.savefig('./json_file_counts.png')
    # plt.show()
    print(df['Folder Name'])



def save_to_csv(data, filename, title="acc"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(['Discipline Name', title])
        writer.writerows(data)

def plot_json_disciline(csv_filename, lim_overall=(90,98), lim_acc=(0,40), title='External'):
    # 读取 CSV 文件
    df = pd.read_csv(csv_filename)
    new_halluc_score, new_title_score = {}, {}
    for key in discipline.keys():
        new_halluc_score[key] = []
        new_title_score[key] = []
    

    for i in range(len(df['Journal Name'])):
        for key in discipline.keys():
            if df['Journal Name'][i] in discipline[key]:
                new_halluc_score[key].append(float(df['Overall hallu score'][i]))
                new_title_score[key].append(float(df['Title acc'][i]))
    # print(new_halluc_score, new_title_score)
    # exit()
    '''hulla score'''
    plt.figure(figsize=(10, 6))
    disc, count = [],[]
    disc_count = []
    for key in new_halluc_score.keys():
        disc.append(key)
        count.append(sum(new_halluc_score[key])/len(new_halluc_score[key])*100)
        disc_count.append((key, sum(new_halluc_score[key])/len(new_halluc_score[key])*100))

    save_to_csv(disc_count, csv_filename.replace("hallu_score", "dicspline_hallu_score"))

    plt.bar(disc, count, color='skyblue')
    plt.xlabel('Discipline')
    plt.ylabel(title + ' Hallucination Score')
    plt.title(title +' Hallucination Score of Each Discipline')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.ylim(lim_overall)

    # 保存图形为文件

    plt.savefig(csv_filename.replace('.csv',"_disciline_ha_score.png"), title='hallu score', )
    
    '''acc'''
    '''hulla score'''
    plt.figure(figsize=(10, 6))
    disc, count = [],[]
    disc_count = []
    for key in new_title_score.keys():
        disc.append(key)
        count.append(sum(new_title_score[key])/len(new_title_score[key])*100)
        disc_count.append((key, sum(new_title_score[key])/len(new_title_score[key])*100))
    
    save_to_csv(disc_count, csv_filename.replace("hallu_score","dicspline_t_score"), title="acc score")

    plt.bar(disc, count, color='green')#,hatch='*'
    plt.xlabel('Discipline')
    plt.ylabel(title + ' Title Accuracy')
    plt.title(title + ' Title Accuracy of Each Discipline')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.ylim(lim_acc)

    # 保存图形为文件

    plt.savefig(csv_filename.replace('.csv',"_disciline_title_acc_score.png"))
    # plt.show()


def plot_radar_chart(model_results_directory='./final_score_t1', title='External'):
    # 读取模型文件夹
    modelnames = [f.name for f in os.scandir(model_results_directory) if f.is_dir()]
    # discipline = {
    #     # 假设discipline是一个字典，示例如下
    #     "Science": ["Journal A", "Journal B"],
    #     "Math": ["Journal C", "Journal D"],
    #     "History": ["Journal E", "Journal F"],
    #     "Literature": ["Journal G", "Journal H"]
    # }
    new_halluc_score = {key: {} for key in discipline.keys()}
    
    # 读取每个模型的分数
    for model in modelnames:
        csv_filename = os.path.join(model_results_directory, model, 'hallu_score.csv')
        df = pd.read_csv(csv_filename)
        for i in range(len(df['Journal Name'])):
            for key in discipline.keys():
                if df['Journal Name'][i] in discipline[key]:
                    if model not in new_halluc_score[key]:
                        new_halluc_score[key][model] = []
                    new_halluc_score[key][model].append(float(df['Overall hallu score'][i]))

    # 计算平均分数
    disc_scores = {key: {} for key in new_halluc_score.keys()}
    for key in new_halluc_score:
        for model in new_halluc_score[key]:
            disc_scores[key][model] = np.mean(new_halluc_score[key][model])
    # print(disc_scores)
    # exit()
    # 整理数据为雷达图形式
    categories = list(discipline.keys())
    num_categories = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    # 绘制雷达图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    for model in modelnames:
        scores = [disc_scores[cat].get(model, 0) for cat in categories]  # 获取模型在每个学科的分数
        scores += scores[:1]  # 闭合数据
        if model == "meta-llama_Llama-3.2-3B-Instruct":
            model = "Llama-3.2"
        if model.startswith("Qwen"):
            model = "Qwen-2.5"
        ax.plot(angles, scores, label=model, linewidth=2)
        ax.fill(angles, scores, alpha=0.25)

    # 添加标签
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, size=20, weight='bold')
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # 保存图像
    plt.tight_layout()
    plt.savefig("./final_score_t1/hallu_score_radar.png", dpi=300)
    # plt.show()

def plot_json_disciline_all_models(model_resutls_dictionary='./final_score_t1', lim_overall=(0,1), lim_acc=(0,40), title='External'):
    # 读取 CSV 文件
        # 设置字体
    # rc('font', family='Times New Roman')

    # 读取 CSV 文件
    modelnames = [f.name for f in os.scandir(model_resutls_dictionary) if f.is_dir()]
    new_halluc_score, new_title_score = {}, {}
    for key in discipline.keys():
        new_halluc_score[key] = {}
        new_title_score[key] = {}
    for model in modelnames:
        csv_filename = os.path.join(model_resutls_dictionary, model, 'hallu_score.csv')
        df = pd.read_csv(csv_filename)
        for i in range(len(df['Journal Name'])):
            for key in discipline.keys():
                if df['Journal Name'][i] in discipline[key]:
                    if model not in new_halluc_score[key].keys():
                        new_halluc_score[key][model] = []
                        new_title_score[key][model] = []
                    new_halluc_score[key][model].append(float(df['Overall hallu score'][i])*100)
                    new_title_score[key][model].append(float(df['Title acc'][i]))

    '''hulla score'''
    # plt.figure(figsize=(12, 8))  # 调整整体图形大小
    disc, count = [], []
    disc_count = []
    for key in new_halluc_score.keys():
        disc.append(key)
        for k in new_halluc_score[key]:
            count.append(sum(new_halluc_score[key][k]) / len(new_halluc_score[key][k]) )
            disc_count.append((key, k, sum(new_halluc_score[key][k]) / len(new_halluc_score[key][k])))
    print(disc_count)
    df = pd.DataFrame(disc_count, columns=['Despline', 'Model', 'Score'])

    # 透视数据，便于绘图
    pivot_df = df.pivot(index='Despline', columns='Model', values='Score')
    journal_names = pivot_df.index
    models = pivot_df.columns
    scores = pivot_df.values
    # print(scores)
    # print(models)
    # exit()
    # 设置柱状图的宽度和位置
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

    # color = ["#FEE066", "#EC817E","#AD0B08","#71BFB2","#237B9F"]
    # color = ["#F3A59A", "#FFC839", "#80D0C3", "#A6DDEA", "#C3EAB5"]
    # claude_color = '#80D0C3'  # '#75C298'  # Blue
    # llama_color = "#F9CDBF" # Orange
    # deepseek_color = "#F3A59A"
    # gpt4_color = '#A6DDEA'     # Green
    # qwen_color = "#9EAAC4"    # Red
    for i in range(len(models)):
        adjusted_x = x + i * (width + gap)  # 添加间距调整

        ax.bar(adjusted_x, df_h[models[i]], width, label=models[i], color=color[i],edgecolor='gray', linewidth=1.0)

    # 添加标签和标题
    # ax.set_xlabel('Discipline', fontsize=25)
    ax.set_ylabel('Accuracy', fontsize=30)
    # ax.set_title('Model Performance Across Different Discipline Categories', fontsize=)
    ax.set_xticks(x + (len(models) - 1) * (width + gap) / 2)  # 调整x轴刻度位置
    ax.set_xticklabels(journal_names, rotation=30, ha='center', fontsize=25)
    ax.tick_params(axis='y', labelsize=25)  # 调整Y轴刻度字体大小

    ax.legend(loc='upper center',fontsize=20, bbox_to_anchor=(0.5,1.2), ncol=5)#bbox_to_anchor=(0.5, -0.2)
    
    # 调整整体布局和字体大小
    plt.tight_layout()
    plt.savefig(model_resutls_dictionary+"acc_score.png", dpi=300)




    
# 测试
if __name__ == "__main__":
    plot_json_disciline('./count_paper.csv')
# plot_json_counts('./count_paper.csv')
