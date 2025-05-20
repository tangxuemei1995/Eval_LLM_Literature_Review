''''
Calculate the entailment relationships between the summaries in Task 2 using the TRUE tool,
which was previously deployed on AutoDL.
The computation results have been downloaded to the t2_turescore and true folders.
'''
import matplotlib.pyplot as plt
import numpy as np
import os,json
import pandas as pd
from matplotlib import rc
from scipy import stats

def read_json_file(filepath):
    """
    读取 JSON 文件并返回内容
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def find_json_files(directory):
    """
    在指定目录中找到所有以 .json 结尾的文件
    """
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

# 数据准备
discipline = {
    "Biology": [ "ento","nutr","ecolsys","animal","phyto","virology","physiol","biophys","micro","cellbio", "arplant", "genom", "genet","neuro","cancerbio","immunol","pharmtox","pathmechdis","med","vision","clinpsy"],
    "Mathematics": [ "statistics",  "biodatasci"],
    # "Psychology": ["devpsych", "clinpsy", "psych", "orgpsych","neuro","vision"],
    "Physics": ["nucl","astro","control","fluid","conmatphys","marine"],
    "Chemistry": ["chembioeng", "bioeng","biochem", "physchem","anchem"],
    # "Clinic": ["cancerbio","immunol","pharmtox","pathmechdis","med","vision","clinpsy"],
    "Social Science": ["criminol","devpsych","psych", "orgpsych","financial","economics","resource","publhealth","soc", "lawsocsci","polisci","linguistics","anthro","crimino"],
    # "Animal and Planet": ["earth","phyto","ento","virology","physiol","animal"],
    "Technology": ["matsci","food","energy","earth"],
}

def relitu():

    models = ['GPT-4o', 'Claude-3.5', 'Qwen-2.5-72B', 'Llama-3.2-3B']  # 模型名称
    categories = ['Entailment', 'Contradiction', 'Neutrality']  # 类别
    data = np.array([
        [1063, 8, 34],  # GPT-4o 数据
        [1068, 10, 27],  # Claude-3.5 数据
        [1048, 8, 48],  # Qwen-2.5-72B 数据
        [1015, 14, 76]  # Llama-3.2-3B 数据
    ])

    # 计算占比
    data_percentage = data / data.sum(axis=1, keepdims=True)

    # 热力图颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 不同类别对应颜色

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_positions = np.arange(len(models))  # X轴位置
    bar_width = 0.6  # 柱宽

    # 绘制每部分的纵向柱状图
    for i, category in enumerate(categories):
        bottom = data_percentage[:, :i].sum(axis=1) if i > 0 else 0  # 累计底部高度
        ax.bar(bar_positions, data_percentage[:, i], bottom=bottom, color=colors[i], label=category, edgecolor='black', width=bar_width)

    # 添加标签和样式
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('Proportion', fontsize=14)
    ax.set_title('Model Performance by Entailment, Contradiction, and Neutrality', fontsize=16)
    ax.legend(title='Category', fontsize=12)

    # 显示每个部分的百分比
    for i in range(len(models)):
        for j in range(len(categories)):
            y_pos = data_percentage[i, :j].sum() + data_percentage[i, j] / 2
            x_pos = bar_positions[i]
            ax.text(x_pos, y_pos, f'{data_percentage[i, j] * 100:.1f}%', ha='center', va='center', fontsize=10, color='white')

    plt.tight_layout()
    plt.savefig('stacked_vertical_bar.png', dpi=300)
    plt.show()

def duidietu():
    # 数据
    # rc('font', family='Times New Roman')

    data = {
        'Claude-3.5': {
            'incorrect': {'Biology': 62, 'Mathematics': 14, 'Physics': 24, 'Chemistry': 24, 'Social Science': 104, 'Technology': 14},
            'total': {'Biology': 460, 'Mathematics': 50, 'Physics': 113, 'Chemistry': 90, 'Social Science': 299, 'Technology': 94}
        },
        'DeepSeek-V3': {'incorrect': {'Biology': 68, 'Mathematics': 8, 'Physics': 29, 'Chemistry': 28, 'Social Science': 96, 'Technology': 14}, 
                    'total': {'Biology': 459, 'Mathematics': 50, 'Physics': 113, 'Chemistry': 90, 'Social Science': 299, 'Technology': 94}
        },
        'GPT-4o': {
            'incorrect': {'Biology': 67, 'Mathematics': 11, 'Physics': 33, 'Chemistry': 26,'Social Science': 100, 'Technology': 9},
            'total': {'Biology': 460, 'Mathematics': 50, 'Physics': 113, 'Chemistry': 90, 'Social Science': 299, 'Technology': 94}
        },
        'Qwen-2.5': {
            'incorrect': {'Biology': 89, 'Mathematics': 22, 'Physics': 42, 'Chemistry': 32, 'Social Science':137, 'Technology': 24},
            'total': {'Biology': 460, 'Mathematics': 50, 'Physics': 113, 'Chemistry': 90, 'Social Science': 299, 'Technology': 94}
        },
        'Llama-3.2': {
            'incorrect': {'Biology': 117, 'Mathematics': 19, 'Physics': 49, 'Chemistry': 42, 'Social Science': 168, 'Technology': 26},
            'total': {'Biology': 460, 'Mathematics': 50, 'Physics': 113, 'Chemistry': 90, 'Social Science': 299, 'Technology': 94}
        },
        
    }

    # 计算正确和错误的占比
    models = ['Claude-3.5', 'DeepSeek-V3','GPT-4o', 'Qwen-2.5', 'Llama-3.2']
    fields = ['Biology', 'Chemistry','Mathematics','Physics','Social Science', 'Technology']

    # 创建数据框
    correct_data = {model: {field: (1 - data[model]['incorrect'][field] / data[model]['total'][field])*100 for field in fields} for model in models}
    incorrect_data = {model: {field: (data[model]['incorrect'][field] / data[model]['total'][field])*100 for field in fields} for model in models}

    df_correct = pd.DataFrame(correct_data, index=fields)
    df_incorrect = pd.DataFrame(incorrect_data, index=fields)

    # 绘制堆叠柱状图
    fig, ax = plt.subplots(figsize=(16, 9))

    # 为每个学科绘制柱状图
    bar_width = 0.14  # 每个柱的宽度
    gap = 0.008        # 组内柱之间的间距

    # 调整x轴的位置，使柱状图之间有间距
    indices = np.arange(len(fields))

    # 先绘制 incorrect 数据
    color = ['#EF666E', 'orange', '#6DCA97', '#9BABD5', '#D0CADE']

    # color = ["#F3A59A", "#FFC839", "#80D0C3", "#A6DDEA", "#C3EAB5"]

    # for i, model in enumerate(models):
    #     adjusted_x = indices + i * (bar_width + gap)  # 添加间距调整
    #     ax.bar(adjusted_x, df_incorrect[model], bar_width, label=f'{model} (Incorrect)', color=color[i])

    # 再绘制 correct 数据（堆叠在 incorrect 数据之上）
    for i, model in enumerate(models):
        adjusted_x = indices + i * (bar_width + gap)  # 添加间距调整
        ax.bar(adjusted_x, df_correct[model], bar_width, label=model, color=color[i],edgecolor='gray', linewidth=1.0)#,hatch='o', linewidth=1)

    # 设置图表标题和标签
    # ax.set_title('Model Performance by Field (Correct and Incorrect Proportions)', fontsize=16)
    # ax.set_xlabel('Fields', fontsize=12)
    ax.set_ylabel('Entail. Score', fontsize=30)

    # 调整x轴刻度的位置
    ax.set_xticks(indices + (len(models) - 1) * (bar_width + gap) / 2)  # 调整x轴刻度位置
    ax.set_xticklabels(fields, rotation=30, ha='center', fontsize=25)
    # ax.tick_params(axis='y', labelsize=25)  # 调整Y轴刻度字体大小
    # ax.legend(loc='upper center',fontsize=25, bbox_to_anchor=(0.5, -0.1),ncol=4)

    # ax.set_xticks(x + (len(models) - 1) * (width + gap) / 2)  # 调整x轴刻度位置
    # ax.set_xticklabels(journal_names, rotation=30, ha='center', fontsize=25)
    ax.tick_params(axis='y', labelsize=25)  # 调整Y轴刻度字体大小

    ax.legend(loc='upper center',fontsize=20, bbox_to_anchor=(0.5,1.2), ncol=5)#bbox_to_anchor=(0.5, -0.2)



    # 显示图例
    # plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.legend(loc='upper center',fontsize=25, bbox_to_anchor=(0.5, -0.1),ncol=4)

    # 显示图表
    plt.ylim((40,95))
    plt.tight_layout()

    # 保存图表
    plt.savefig('./figures/t2_true_entailment.png')
    # plt.show()

def count_t2_true():
    model2score = {}
    for llm in ["deepseek","meta-llama_Llama-3.2-3B-Instruct", "Qwen_Qwen2.5-72B-Instruct","gpt-4o-new","claude"]:
        # model2score = {}
        with open('./discipline.json', 'r', encoding='utf-8') as f:
            discipline = json.load(f)
        discipline2acc = {}
        topics = [f.name for f in os.scandir("./t2_truescore/" + llm + '/data/') if f.is_dir()]
        no_entail_count, dis_count = {}, {}
        for key in discipline.keys():
            no_entail_count[key] = 0
            dis_count[key] = 0
        model_nli_list = []
        for t in topics:
            if t=="arcompsci":
                continue
            journal_llm_path = './t2_truescore/'+ llm + '/data/' + t

            subdirectories = [d for d in os.listdir(journal_llm_path) if os.path.isdir(os.path.join(journal_llm_path, d))]
            journal_entail, journal_count = 0,0
            for subd in subdirectories:
                json_files = find_json_files(os.path.join(journal_llm_path, subd))
                # print(t, len(json_files))
                for file in json_files:
                    text = read_json_file(file)
                    flag = 0
                    journal_count += 1
                    for key in discipline.keys():
                        if t in discipline[key]: #pharmtox 30
                            flag = 1
                            if text["label"] != 1:
                                no_entail_count[key] += 1
                            dis_count[key] += 1
                            if text["label"] == 1:
                                journal_entail += 1
                    if flag == 0:
                        print(t)

            nli_score_journal = journal_entail/journal_count
            model_nli_list.append(nli_score_journal)
            for key in discipline.keys():
                if t in discipline[key]:
                    if key not in discipline2acc.keys():
                        discipline2acc[key] = [nli_score_journal]
                    else:
                        discipline2acc[key].append(nli_score_journal)
        # print(discipline2acc) 
        
        biology_data = discipline2acc["Biology"]
        chemistry_data = discipline2acc["Chemistry"]
        math_data = discipline2acc["Mathematics"]
        physics_data = discipline2acc["Physics"]
        sociology_data = discipline2acc["Social Science"]
        f_stat, p_val = stats.f_oneway(biology_data, chemistry_data, math_data, physics_data, sociology_data)
        
        print(llm, f"ANOVA: F-statistic = {f_stat}, P-value = {p_val}")
        anova_path = os.path.join('./t2_truescore' + '/', llm, 'anova.txt')
        with open(anova_path, 'w', encoding='utf-8') as w:
            w.write(f"ANOVA: F-statistic = {f_stat}, P-value = {p_val}")

        model2score[llm] = {"incorrect":no_entail_count,"total": dis_count}

        print("entail:", sum(model_nli_list)/len(model_nli_list))
        # print()    
    return model2score
if __name__=="__main__":
    # model2score = count_t2_true()
    # print(model2score)
    duidietu() #绘制第二个任务的柱状图

'''Journal: cellbio, number of article: 16
Journal: statistics, number of article: 28
Journal: devpsych, number of article: 18
Journal: anthro, number of article: 19
Journal: micro, number of article: 30
Journal: nucl, number of article: 17
Journal: chembioeng, number of article: 13
Journal: arplant, number of article: 28
Journal: genom, number of article: 17
Journal: clinpsy, number of article: 19
Journal: biochem, number of article: 17
Journal: financial, number of article: 31
Journal: ento, number of article: 23
Journal: marine, number of article: 21
Journal: nutr, number of article: 17
Journal: bioeng, number of article: 17
Journal: energy, number of article: 30
Journal: anchem, number of article: 20
Journal: soc, number of article: 21
Journal: conmatphys, number of article: 18
Journal: vision, number of article: 23
Journal: fluid, number of article: 26
Journal: publhealth, number of article: 22
Journal: psych, number of article: 22
Journal: pathmechdis, number of article: 21
Journal: criminol, number of article: 22
Journal: biophys, number of article: 25
Journal: orgpsych, number of article: 20
Journal: earth, number of article: 25
Journal: matsci, number of article: 16
Journal: physchem, number of article: 23
Journal: neuro, number of article: 20
Journal: food, number of article: 23
Journal: linguistics, number of article: 29
Journal: lawsocsci, number of article: 20
Journal: economics, number of article: 27
Journal: animal, number of article: 15
Journal: cancerbio, number of article: 18
Journal: genet, number of article: 18
Journal: biodatasci, number of article: 22
Journal: virology, number of article: 22
Journal: astro, number of article: 13
Journal: ecolsys, number of article: 22
Journal: physiol, number of article: 21
Journal: resource, number of article: 23
Journal: phyto, number of article: 18
Journal: polisci, number of article: 25
Journal: pharmtox, number of article: 31
Journal: control, number of article: 18
Journal: immunol, number of article: 22
Journal: med, number of article: 34'''