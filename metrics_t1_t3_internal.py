
'''
Calculate the overlap between LLM-generated reference and original references in Task 1 and Task 3 using evaluate_internal_t1.py
'''
import os
from llm_api import find_json_files, read_json_file
from semantic_scholar import semantic_scholar_search
import csv

from figure import plot_json_disciline,plot_json_disciline_all_models


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




if __name__ == "__main__":

    tasks = ["t1", "t3"]
    # with open('./discipline.json', 'r', encoding='utf-8') as f:
    #     discipline = json.load(f)
    # discipline2acc = {}
    for task in tasks:
        print("_______________________", task, "___________________________")
        llms = ["deepseek"]#, "Qwen_Qwen2.5-72B-Instruct","claude","gpt-4o-new","meta-llama_Llama-3.2-3B-Instruct"]
        first_authors = [True, False]
        for first_author in first_authors:
            for llm in llms:
                topics = [f.name for f in os.scandir( './compare_original_data_' + task + '/'+ llm  +'/data/') if f.is_dir()]

                
                all_correct = {"title_correct":0, "author_correct":0, "journal_correct":0, "year_correct":0,"volume_correct":0, "first_page_correct":0, "last_page_correct":0} 
                all_count = {"title_correct":0, "author_correct":0, "journal_correct":0, "year_correct":0,"volume_correct":0, "first_page_correct":0, "last_page_correct":0} 
                all_correct_no_hallu = {"first_author":0, "title_correct":0, "author_correct":0, "journal_correct":0, "year_correct":0,"volume_correct":0, "first_page_correct":0, "last_page_correct":0} 

                sum_, title_acc_, title_fuzzy_, title_acc_other, title_fuzzy_other = 0, 0, 0, 0, 0
                search = 0
                title_80 = []
                if not os.path.exists('./final_score_internal_' + task + '/'+ llm):
                    os.makedirs('./final_score_internal_' + task + '/'+ llm)
                save_path = os.path.join('./final_score_internal_' + task + '/', llm, 'overlap_score.csv')
                true_list = []
                no_hallu_paper_count = 0
                title_fuzzy_acc, halluc_score_model = [], []
                journal_hallu_list = []
                for t in topics[0:52]:
                    journal_llm_path = './compare_original_data_' + task + '/'+ llm  + '/data/' + t
                    sum_r_journal, refer_right_journal, title_right_journal = 0, 0, 0
                    subdirectories = [d for d in os.listdir(journal_llm_path) if os.path.isdir(os.path.join(journal_llm_path, d))]
                    for subd in subdirectories:
                        #2023年这一期
                        json_files = find_json_files(os.path.join(journal_llm_path, subd))
                        title_correct_journal, halluc_score_journal = [], []
                        for f in json_files:
                            #某一篇文章对应的幻觉
                            sum_r_paper, refer_right_paper, title_right_paper = 0, 0, 0
                            text = read_json_file(f)
                            article_score, artcle_all_score = 0.0, 0.0
                            for z in text: 
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
                                # search += z["search_score"]
                                title_related_count, pro_related_count = calculate_scores(list_score=list_score, list_score_pro=list_score_pro, first_author=first_author)
                                
                                title_acc_ += title_related_count #title 完全正确，且有其他一项正确
                                refer_right_journal += title_related_count
                                refer_right_paper += title_related_count

                                title_fuzzy_ += pro_related_count #title 80%完全正确，且有其他一项正确
                                refer_right_journal += pro_related_count
                                refer_right_paper += pro_related_count
                                #title 不对，但其他有三项对
                                title_related_count_, pro_related_count_ = calculate_scores_(list_score=list_score, list_score_pro=list_score_pro, first_author=first_author)
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
                                    
                        # no_hallu_paper_count += refer_right_journal 

                            # title_80 += title_right_journal
                    # with open('./final_score_t1/'+ llm +'/hallu_score.txt','w', encoding='utf-8') as writer:
                    # title_80.append(title_right_journal/sum_r_journal)
                            title_fuzzy_acc.append(title_right_paper/sum_r_paper) #100%-80%的标题正确的准确率，
                            acc_paper = (refer_right_paper/sum_r_paper) #每篇文章的幻觉
                            title_correct_journal.append(title_right_paper/sum_r_paper)
                            halluc_score_journal.append(acc_paper)  #幻觉
                            halluc_score_model.append(acc_paper)
                            # for key in discipline.keys():
                            #     if t in discipline[key]:
                            #         if key not in discipline2acc.keys():
                            #             discipline2acc[key] = [acc_paper]
                            #         else:
                            #             discipline2acc[key].append([acc_paper])
                        journal_hallu_list.append((t, sum(title_correct_journal)/len(title_correct_journal) ,(sum(halluc_score_journal)/len(halluc_score_journal))))
                save_to_csv(journal_hallu_list, save_path)
                # print(len(halluc_score_model))
                # exit()
                print("##########################", llm, "first_author", first_author,"#########################")
                if first_author:
                    save_path_1 = os.path.join('./final_score_internal_' + task + '/', llm, '其他统计结果（first author）.txt')
                else:
                    save_path_1 = os.path.join('./final_score_internal_' + task + '/', llm, '其他统计结果.txt')

                with open(save_path_1, 'w', encoding='utf-8') as f:
                    print("检索率:",search/sum_)

                    print("生成的文献总数：", sum_)    
                    f.write("生成的文献总数：\t" + str(sum_) + '\n')
                    print("生成的文献总数：", sum_), 

                # print("title不正确，但是其他三项正确的占比", (title_acc_other+title_fuzzy_other)/sum_)
                    print("title 80-100acc", sum(title_fuzzy_acc)/len(title_fuzzy_acc))
                    print("average acc", sum(halluc_score_model)/len(halluc_score_model))
                    print("分子每个类别中正确的项，分母是所有生成的该项")

                    f.write("title 80-100acc" + str(sum(title_fuzzy_acc)/len(title_fuzzy_acc)) +'\n')
                    f.write("average acc" +str(sum(halluc_score_model)/len(halluc_score_model)) + '\n')
                    # print("分子每个类别中正确的项，分母是所有生成的该项")
                    # f.write("分子每个类别中正确的项，分母是所有生成的该项\n")

                    # for key in all_correct.keys():
                    #     print(key, all_correct[key]/all_count[key])
                    #     f.write(key + '\t' + str(all_correct[key]/all_count[key]) + '\n' )
                    print("分子是no hallucination paper中每个类别中正确的项，分母是所有生成的该项")
                    f.write("分子是no hallucination paper中每个类别中正确的项，分母是所有生成的该项\n")

                    for key in all_correct_no_hallu.keys():
                        if key != "first_author":
                            print(key, all_correct_no_hallu[key]/all_count[key])
                            f.write(key + '\t' + str(all_correct_no_hallu[key]/all_count[key]) + '\n' )
                        else:
                            print(key, all_correct_no_hallu[key]/all_count["author_correct"])
                            f.write(key + '\t' + str(all_correct_no_hallu[key]/all_count["author_correct"]) + '\n' )
            