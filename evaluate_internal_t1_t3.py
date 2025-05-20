'''
Compare the OVERLAP between references and original sources in Task 1 and Task 3,  
primarily to generate data in `./compare_original_data_t3/t1/` for computation in `metrics_t1_t3.py`.
'''
from llm_api import find_json_files, read_json_file
import os
import json
import re
import time
import csv
from semantic_scholar import semantic_scholar_search
# import litellm
import openai
# from litellm import completion 
from compare_author import compare_author_lists_for_t1_internal
from compare_journal import compare_journals
"""
evaluation for references and 
"""
openai.api_key = "your openai key" 
f = open('./prompt3-gpt.txt', 'r', encoding='utf-8')
# f.read()
system_prompt = f.read().strip()

#
def gpt(system_prompt, user_prompt):
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

def clean_references(json_files):

    for file in json_files:
        # print("###########################", file, "#########################")
        # exit()
        if os.path.exists("./clean/" + file):
            continue
        with open(file, 'r', encoding='utf-8') as f:
            # data_dict = json.load(file)
            text = f.read().replace("}\n\n{",",")
        try:
            print("###########################", file, "#########################")
            llm_generated = json.loads(text)
            # if type(llm_generated) == "dict":
            data_dict = llm_generated
            #     print("read it as json!", file)
            
        #     else:
        #         print("1: can't read the file as json, so use gpt to convert it")
        #         initial_json_file = file.replace("./llm3/gpt-3.5","clean")
        #         data = read_json_file(initial_json_file)
        #         # print(json_file)
        #         user_prompt = "ABSTRACT:" + data["context"]["ABSTRACT"]
                
        #         llm_generated= gpt(system_prompt, user_prompt)
        #         print(llm_generated)
        #         # try:
        #         data_dict = json.loads(llm_generated)
        # except:
        #     print("###########################", file, "#########################")
        #     exit()

        # print(type(data_dict))
        # exit()
        except:
            
            print("1: can't read the file as json, so use gpt to convert it")
            system_prompt = "please help me to modify the next text, modify notations and make it become a normal json file, the file only includes one dict, and 'Literature Review' and 'References' are two keys, and the value of 'References' is a list, the list includes several dict, each dict's keys are: 'title', 'authors', 'journal','year','volumes','first page' and 'last page', if the dict doesn't include all keys, please delete the dict, please only output the json file:"
            # user_prompt = "please help me to modify the next text, make it become a normal json file,only output the json file:" + text
            user_prompt =  text
            # print(text)
            llm_generated= gpt(system_prompt, user_prompt)
            print("######################convert it as json",llm_generated,"##########################") 
        try:
            if isinstance(llm_generated, str):
                # print(llm_generated)
                # print(type(llm_generated))
                data_dict = json.loads(llm_generated)
            else:
                data_dict = llm_generated
        except:
            print("2: can't read the file as json, so use gpt to convert it")
            system_prompt = "please help me to modify the next text, modify notations and make it become a normal json file, the file only includes one dict, and 'Literature Review' and 'References' are two keys, and the value of 'References' is a list, the list includes several dict, each dict's keys are: 'title', 'authors', 'journal','year','volumes','first page' and 'last page', if the dict doesn't include all keys, please delete the dict, please only output the json file:"
            user_prompt =  llm_generated
            # user_prompt = "please help me to modify the next text, don't delete any content, only modify notations and make it become a normal json file, the file only includes one dict, and 'References' is the dict key, and the value of 'References' is a list, the list includes several dict, each dict's keys are: 'title', 'authors', 'journal','year','volumes','first page' and 'last page', please only output the json file:" + llm_generated
            llm_generated = gpt(system_prompt, user_prompt)

            # print(llm_generated)
            # exit()
            if isinstance(llm_generated, str):
                # print(file)
                # print("###########################")

                # print(llm_generated)
                f = open('./test.txt','w',encoding='utf-8')
                f.write(llm_generated)
                # print("###########################")
                try:
                    # print([error_position-20:error_position+20])
                    data_dict = json.loads(llm_generated.replace("}\n\n{","},{").replace("success callbacks: []","").replace("\n\n","\n").replace("\\",""), strict=False)
                except:
                    data_dict = json.loads(llm_generated)
            else:
                data_dict = llm_generated

        with open("./clean/" + file, 'w', encoding='utf-8') as f3:
            json.dump(data_dict, f3, ensure_ascii=False)
        # exit()                      
        # print(type(data_dict)) 
        # print(llm_generated["References"])

def compare_papers(paper, results):
    '''compare the paper generated by LLM and its candidates from google scholar'''
    # title = paper["title"].strip().lower()

    def normalize_title(title):
        return title.strip().lower()

    # 标准化作者，用于比较
    def normalize_authors(authors):
        return authors.strip().lower()

    # 标准化页码
    def normalize_pages(page):
        return str(page).strip()
    
    def parse_apa_content(apa_content, gbt_content):
        # 匹配作者
        print("apa_content", apa_content)
        apa_content = apa_content.replace('. (Eds.)', '')
        authors = apa_content.split(". (")[0].replace('&', '')
        authors =[normalize_authors(a.strip()) for a in authors.split("., ")]
        text = apa_content.split(".")[-2]
        print(gbt_content)

        try:
            # text = text.split(').')[1].split('.')[1]
            # print(text)
            # exit()
            text = text.split(',')
            if len(text) == 3:
                journal, volumes, pages = text[0].strip(), text[1].strip(), text[2].strip()
            elif len(text) == 2:
                journal, volumes, pages = text[0].strip(), text[1].strip(), ""
            elif len(text) == 1:
                journal, volumes, pages = text[0].strip(), "", ""
            else:
                journal, volumes, pages = "", "", ""
        except:
            # print("some thing wrong in this apa", apa_content)
            # print(text, authors)
            journal, volumes, pages = "", "", ""
            exit()

    
        # print(authors, journal, volumes, pages)
        # exit()
        return {
            'authors': authors,
            'journal': journal,
            'volumes': volumes,
            'pages': pages,
            # 'year':year
        }



    paper_title = normalize_title(paper['title'])
    # print("paper authors", paper['authors'])
    paper_authors = [normalize_authors(a.strip()) for a in paper['authors'].strip().replace("and", "").replace("&","").split(".,")]
    paper_year = paper['year'].strip()
    if 'volumes' not in paper.keys():
        paper['volumes'] = ""
    paper_volumes = paper['volumes'].strip()
    if 'first page' not in paper.keys():
        paper['first page'] = ""
    paper_first_page = normalize_pages(paper['first page'])
    if 'last page' not in paper.keys():
        paper['last page'] = ""
    paper_last_page = normalize_pages(paper['last page'])
    paper_journal = normalize_title(paper['journal'])
    compare_list = []
    title_correct = 0
    for result in results:
        info_list = [] #共7项，只有当title满足时在判断其他
        result_title = normalize_title(result['title'])
        result_year = result['year']
        if result_year == -1:
            try:
                result_year = re.findall(r'\(\d{4}\)',  result['apa_content'])[0]
            except:
                try:
                    result_year = re.findall(r', \d{4},',  result['gbt_content'])[0]
                except:
                    result_year = ""
        
        # print("result_year", result_year)
        # print()
        # result_authors = [normalize_authors(a.strip()) for a in result['authors'].strip().replace("and", "").split(",")]
        # result_year = parsed_results['year']
        parsed_results = parse_apa_content(result['apa_content'], result['gbt_content'])
        # result_year = parsed_results['year']
        result_authors = parsed_results["authors"]
        # print(parsed_results)
        result_volumes = parsed_results["volumes"]
        if "(" in result_volumes:
            result_volumes = [x.replace(")", "") for x in result_volumes.split('(')]
        else:
            result_volumes = [result_volumes]
        result_page = parsed_results["pages"]
        if result_page != "":
            if "-" in result_page:
                result_first_page, result_last_page = result_page.replace('.','').split("-")
            else:
                result_first_page, result_last_page = result_page, ""
        else:
            result_first_page, result_last_page = "", ""
        result_journal = parsed_results['journal'].lower()
        result_first_page, result_last_page = normalize_pages(result_first_page), normalize_pages(result_last_page)


        apa = [result_authors, result_journal, result_title, result_year, result_volumes, result_first_page, result_last_page]
        paper = [paper_authors, paper_journal, paper_title, paper_year, paper_volumes, paper_first_page, paper_last_page]
        
        # print("paper information:", paper)
        # print("candidate information:", apa)
        # exit()
        if paper_title == result_title:
        
            print(paper_title,"#####yes this paper title is correct!######")
            info_list.append(1)
            # exit()
            if paper_authors == result_authors:
                info_list.append(1)
            else:
                info_list.append(0)
            if paper_year == result_year:
                info_list.append(1)
            else:
                info_list.append(0)
            if paper_volumes in result_volumes:
                info_list.append(1)
            else:
                info_list.append(0)
            if paper_first_page == result_first_page:
                info_list.append(1)
            else:
                info_list.append(0)
            if paper_last_page == result_last_page:
                info_list.append(1)
            else:
                info_list.append(0)
        else:
            info_list.append(0)
        if 1 in info_list:
            compare_list.append(info_list)
    
    if compare_list != []: #说明LLM生成的文献存在
        # print(compare_list)
        # exit()
        title_correct += 1
        if len(compare_list) != 1:
            score = 0
            for x in compare_list:
                if sum(x) > score:
                    score = sum(x)
        else:
            score = sum(compare_list[0])
    else:
        score = 0
    return score, title_correct




def average(aver):
    total_sum = sum(aver)
    average_score = total_sum / len(aver)
    return average_score


def remove_special_characters(text):
    # 使用正则表达式替换所有非字母的字符（保留字母）
        cleaned_text = re.sub(r'[^A-Za-z]', ' ', text)
        cleaned_text = cleaned_text.replace("  ", " ")
        return cleaned_text.strip()


def compute_metrics(candidate, paper_info):

    ''''''
    # def remove_special_characters(text):
    # # 使用正则表达式替换所有非字母的字符（保留字母）
    #     cleaned_text = re.sub(r'[^A-Za-z]', ' ', text)
    #     cleaned_text = cleaned_text.replace("  ", " ")
    #     return cleaned_text.strip()

    # print("begin to compute the metric")
    # print(remove_special_characters(candidate["title"]))
    # print(remove_special_characters(paper_info["title"]))
    candidate["title"] = remove_special_characters(candidate["title"] )
    paper_info["title"] = remove_special_characters(paper_info["title"])
    title_correct, author_correct, journal_correct, year_correct, volume_correct, first_page_correct, last_page_correct = 0,0,0,0,0,0,0
    
    if candidate["title"] == paper_info["title"]:
        print(paper_info["title"],"#####yes this paper title is correct!######")
        title_correct = 1

        if candidate["author"] == paper_info["author"]: #还需要细致的比较
            author_correct = 1

        if candidate["journal"] == paper_info["journal"]:
            journal_correct = 1

        if candidate["year"] == paper_info["year"]:
            year_correct = 1

        if candidate["volume"] == paper_info["volume"]:
            volume_correct = 1

        if candidate["first_page"] == paper_info["first_page"]:
            first_page_correct = 1

        if candidate["last_page"] == paper_info["last_page"]:
            last_page_correct = 1
    
        
    return [title_correct, author_correct, journal_correct, year_correct, volume_correct, first_page_correct, last_page_correct]

def compare_title_strings(str1, str2):
    words1 = set(str1.split())
    words2 = set(str2.split())
    
    overlap = words1.intersection(words2)
    similarity = len(overlap) / len(words1) if words1 else 0
    
    return 1 if similarity >= 0.8  else 0

def compare_author_(author_list1, author_list2):
    author_set = compare_author_lists_for_t1_internal(author_list1, author_list2)

    # if author_set != []:
    #     return 1
    # else:
    #     return 0
    # print("author_set:", author_set)
    # exit()
    # print(author_list2)
    if len(author_set) == len(author_list2):
        return 1
    else:
        return 0
        # print(author_list1)
        # print(author_list2)
        # print(author_set)
        # exit()
def compare_first_author_(author_list1, author_list2):
    '''对比第一作者'''
    author_set = compare_author_lists_for_t1_internal(author_list1, author_list2)
    # if author_set != []:
    #     return 1
    # else:
    #     return 0
    # if len(author_set) == len(author_list2):
    flag = 0
    for x in author_set:
        if author_list2[0] in x:
            flag = 1

    if flag == 0:
        return 0
    else:
        return 1
    
def compute_metrics_fuzzy(candidate, paper_info):
    '''模糊匹配'''
    def remove_special_characters(text):
    # 使用正则表达式替换所有非字母的字符（保留字母）
        cleaned_text = re.sub(r'[^A-Za-z]', ' ', text)
        cleaned_text = cleaned_text.replace("  ", " ")
        return cleaned_text

    candidate["title"] = remove_special_characters(candidate["title"]).strip()

    paper_info["title"] = remove_special_characters(paper_info["title"]).strip()
    # title_correct, author_correct, journal_correct, year_correct, volume_correct, first_page_correct, last_page_correct = 0,0,0,0,0,0,0
    '''title-flag=1 if similarity >=0.8'''
    title_flag = compare_title_strings(paper_info["title"],candidate["title"] )
    '''作者是否完全一致'''
    author_flag = compare_author_(candidate["author"], paper_info["author"])
    paper_dict = {"title_correct": 0, "author_correct": 0, "journal_correct": 0, "year_correct": 0,"volume_correct": 0, "first_page_correct": 0, "last_page_correct": 0} 
    paper_dict_pro = {"title_correct_0.8": 0, "first_author" :0} 
    
    
    if title_flag == 1:
       paper_dict_pro["title_correct_0.8"] = 1
    else:
        if paper_info["title"] == "" or candidate["title"] == "":
            if candidate["title"] == "":
                '''一些book的引用没有title'''
                candidate["title"] = candidate["journal"]  
                title_flag_ = compare_journals(candidate["title"], paper_info["journal"]) 
                if title_flag_ == 1:
                    paper_dict_pro["title_correct_0.8"] = 1
                else:
                    paper_dict_pro["title_correct_0.8"] = 0
            else: #paper_info["title"] == ""这时说明llm未生成title，则赋值-1
                paper_dict_pro["title_correct_0.8"] = -1
        else:
            paper_dict_pro["title_correct_0.8"] = 0

    if paper_info["title"] == candidate["title"]:
        paper_dict["title_correct"] = 1
    else:
        if paper_info["title"] == "" or candidate["title"] == "":
            paper_dict["title_correct"] = -1
        else:
            paper_dict["title_correct"] = 0
    
    first_author_flag = compare_first_author_(candidate["author"], paper_info["author"])

    if first_author_flag == 1: #还需要细致的比较
            paper_dict_pro["first_author"] = 1
    else:
        if paper_info["author"] == [] or candidate["author"] == []:
            paper_dict_pro["first_author"] = -1
        else:
            paper_dict_pro["first_author"] = 0

    if author_flag == 1: #还需要细致的比较
        paper_dict["author_correct"] = 1
    else:
        if paper_info["author"] == [] or candidate["author"] == []:
            paper_dict["author_correct"] = -1
        else:
            paper_dict["author_correct"] = 0

    # print(candidate["journal"])
    # print(paper_info["journal"])
    # exit()
    journal_flag = compare_journals(candidate["journal"], paper_info["journal"])
    
    if journal_flag == 1:
        paper_dict["journal_correct"] = 1
    else:
        if paper_info["journal"] == [] or candidate["journal"] == []:
            paper_dict["journal_correct"] = -1
        else:
            paper_dict["journal_correct"] = 0

    if str(candidate["year"]) == "" or str(paper_info["year"]) == "":
        paper_dict["year_correct"] = -1
    else:
        if str(candidate["year"]) == str(paper_info["year"]):
            paper_dict["year_correct"] = 1
        else:
            paper_dict["year_correct"] = 0


    if str(candidate["volume"]) == "" or str(paper_info["volume"]) == "":
        paper_dict["volume_correct"] = -1
    else:
        if str(candidate["volume"]) == str(paper_info["volume"]):
            paper_dict["volume_correct"] = 1
        else:
            paper_dict["volume_correct"] = 0

    if str(candidate["first_page"]) == "" or str(paper_info["first_page"]) == "":
        paper_dict["first_page_correct"] = -1
    else:
        if str(candidate["first_page"]) == str(paper_info["first_page"]):
            paper_dict["first_page_correct"] = 1
        else:
            paper_dict["first_page_correct"] = 0

    if str(candidate["last_page"]) == "" or str(paper_info["last_page"]) == "":
        paper_dict["last_page_correct"] = -1
    else:
        if str(candidate["last_page"]) == str(paper_info["last_page"]):
            paper_dict["last_page_correct"] = 1
        else:
            paper_dict["last_page_correct"] = 0
    
    # print(paper_dict)
    # exit()
    return paper_dict, paper_dict_pro


def compare_semantic_papers(paper, semantic_results):
    '''Compare the information from LLM generated and search the paper title in semantic scholar'''
    def normalize_title(title):
        return title.strip().lower()

    # 标准化作者，用于比较
    def normalize_authors(authors):
        return authors.strip().lower()

    # 标准化页码
    def normalize_pages(page):
        return str(page).strip()
        
    paper_ = paper
    paper = {}
    for key in paper_.keys():
        paper[key.lower()] = paper_[key]
    paper_title = normalize_title(paper['title'])
    # print("paper", paper)
    if 'journal'not in paper.keys():
        paper['journal'] = ""
    if 'authors'not in paper.keys():
        paper['authors'] = ""
    # print("paper authors", paper['authors'])

    # paper['authors'] = paper['authors'].replace(" et al.","").replace("...","")
    paper_authors = []
    # print(paper['authors'])
    if not isinstance(paper['authors'], list):
        paper['authors'] = paper['authors'].replace(" et al.","").replace("...","")
        paper['authors'] = paper['authors'].strip().replace(" and ", " ., ").replace("&","., ")
        if "." in paper['authors']:
            #有简写
            for a in paper['authors'].split(".,"):
                paper_authors.append(normalize_authors(a.strip()))
            if  len(paper_authors)==1 and len(paper_authors[0])> 20:
                paper_authors = paper_authors[0].split(",")
        else:
            #无简写
            for a in paper['authors'].split(","):
                paper_authors.append(normalize_authors(a.strip()))
    else:

        paper_authors = paper['authors']

    # paper_authors = list(set(paper_authors))
    paper_authors_ = paper_authors
    paper_authors = []
    for item in paper_authors_:
         item = item.strip()
         if item != '' and item not in paper_authors:
             paper_authors.append(item)
    if 'year' not in paper.keys():
        paper["year"] = ""
    # else:
    paper_year = str(paper['year']).strip()
    if 'volumes' not in paper.keys():
        paper['volumes'] = ""
    paper_volume = str(paper['volumes']).strip()
    if 'first page' not in paper.keys():
        paper['first page'] = ""
    paper_first_page = normalize_pages(str(paper['first page']))
    if 'last page' not in paper.keys():
        paper['last page'] = ""
    paper_last_page = normalize_pages(str(paper['last page']))

    paper_journal = normalize_title(paper['journal'])
    paper_info= {"author":paper_authors,\
        "title": paper_title,\
        "volume": str(paper_volume).strip(), \
        "first_page": str(paper_first_page).strip() ,\
        "year": str(paper_year).strip(),\
        "journal": str(paper_journal).strip(),\
        "last_page": str(paper_last_page).strip() }
    # print(paper_title, paper_authors, paper_year, paper_volumes, paper_first_page, paper_last_page, paper_journal)
    # exit()

    '''parse the paper in semantic results'''
    score_list = []
    sum_score = 0
    candidate_list = []
    best_candidate = {"sum_score": 0, "c_score_pro": {"title_correct_0.8":0, "first_author":0} , "score_list":  {"title_correct":0, "author_correct":0, "journal_correct":0, "year_correct":0,"volume_correct":0, "first_page_correct":0, "last_page_correct":0} }
    # print(semantic_results)
    i = 0
    for key in semantic_results.keys():
        item = semantic_results[key]
        c_title = item["title"].lower().replace(".", "")
        c_year = item["year"]
        c_journal = item["source"]
        c_year = item["year"]
        pages = item["pages"]
        c_volume = item["volume"]
        pages = re.sub(r'[a-zA-Z]', '', pages)
        if "." in pages:
            pages = pages.split(".")[0]
        if ":" in pages:
            pages = pages.split(":")[0]
        if "–" in pages:
            print("- in pages", pages)
            c_first_page, c_last_page = pages.split("–")
        else:
            c_first_page, c_last_page = pages, ""
            # print(item)
            print("pleaese, there is no pages", pages)
            # exit()
        
        author_list = item["authors"].split(",")

        candidate_list.append({"author":author_list,\
        "title": c_title.strip(),\
        "volume":str(c_volume).strip(), \
        "first_page":c_first_page.strip(),\
        "year":str(c_year).strip(),\
        "journal":c_journal.strip(),\
        "last_page":str(c_last_page).strip()})

        # print(candidate_list)
        # exit()
        c_score_dict, c_score_pro = compute_metrics_fuzzy(candidate_list[i], paper_info)
        
        c_score_list = []
        for key in c_score_dict.keys():
            if c_score_dict[key] != -1:
                c_score_list.append(c_score_dict[key])

        c_sum_score = sum(c_score_list)/len(c_score_list)

        candidate_list[i]["sum_score"] = c_sum_score
        candidate_list[i]["score_list"] = c_score_dict
        candidate_list[i]["c_score_pro"] = c_score_pro
        candidate_list[i]["search_score"] = 1
        # if candidate_list[i]["score_list"]["title_correct"] ==0 and \
        #      candidate_list[i]["score_list"]["journal_correct"] == 1 \
        #     and candidate_list[i]["score_list"]["author_correct"] == 0\
        #     and candidate_list[i]["score_list"]["year_correct"] == 1 \
        #     and candidate_list[i]["score_list"]["volume_correct"] == 1:
            # print(candidate_list[i])
            # print(paper_info)
            # exit()
        if best_candidate["score_list"]["title_correct"] != 1 and  best_candidate["c_score_pro"]["title_correct_0.8"] != 1:
            #当前还没到title匹配的candidate
            if c_sum_score >= sum_score:
                sum_score = c_sum_score
                best_candidate = candidate_list[i]
        else: #已经找到了title匹配的情况，要看新找到的candidate是否在title也匹配
            #，如果title匹配的同时分数更高则作为best
            if candidate_list[i]["score_list"]["title_correct"]==1 or candidate_list[i]["c_score_pro"]["title_correct_0.8"] ==1:
                if c_sum_score >= sum_score:
                    sum_score = c_sum_score
                    best_candidate = candidate_list[i]
        i += 1
    # print(best_candidate)
    '''
    {'author': ['a. bruu'], 
    'title': 'enteroviruses polioviruses coxsackieviruses echoviruses and newer enteroviruses',
    'volume': '', 
    'first_page': '44', 
    'year': 2003, 
    'journal': '', 
    'last_page': '45', 
    'sum_score': 1, 
    'score_list': [1, 0, 0, 0, 0, 0, 0]}
    '''
    
    
    return best_candidate
 


def compute_overlap(json_files, model_id, task):
    '''
    for each Journal search each reference from google scholar
    '''
    title_aver, info_aver, halluc = [],[],[]
    index = 0 

    for file in json_files:
        print("#########################################" + file + "##############################################" )
        generate = read_json_file(file)
        index += 1
        '''
        save path for the overlap reference
        '''
        save_path = file.replace('./' + task, './compare_original_data_'+ task)
        # print(save_path)
        # exit()
        if not os.path.exists('/'.join(save_path.split('/')[0:-1])):
            os.makedirs('/'.join(save_path.split('/')[0:-1]))
        if task == "t3":
            if "Literature Review" in generate.keys():
                if isinstance(generate["Literature Review"], dict):
                    if "References" in generate["Literature Review"].keys():
                        t, r = "", generate["Literature Review"]["References"]
                else:
                    t, r = generate["Literature Review"], generate["References"]
            else:
                t = ""
                r =  generate["References"]
        elif task == "t1":
            r =  generate["References"]
        info_correct, correct_title, all = [], [], 0
        best_candidate_info = []  #用于将每个llm产生的文章和原来的参考文献进行对比
        # print("generated references", r)

        '''找到原始文章中的references'''
        original_article_path = file.replace("./"+ task +"/" + model_id.replace('/',"-"), "./clean")
        original_reference = read_json_file(original_article_path)["reference"]
        ref_results = {}
        for ref in original_reference:
            if ref["title"].strip() == "":
                ref_results[ref["source"].lower().strip()] = ref
            else:
                ref_results[ref["title"].lower().strip().replace('.', '')] = ref
        # print(ref_results)
        # exit()
        all = 0
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
            all += 1
            candidate = compare_semantic_papers(paper,ref_results)
            # print(candidate)
            best_candidate_info.append(candidate)
            if candidate["score_list"]["title_correct"] == 1:
                correct_title.append(1)
            else:
                correct_title.append(0)
            info_correct.append(candidate["sum_score"])
        # print(save_path)
        # exit()
        with open(save_path, 'w', encoding='utf-8') as f4:
            json.dump(best_candidate_info, f4, ensure_ascii=False)
        print("title_seacrh")
        print("title_correct:", sum(correct_title)/len(correct_title) if correct_title else 0)

        print("info_correct:", sum(info_correct)/len(info_correct) if info_correct else 0)
        print("hullcination info", 1- sum(info_correct)/len(info_correct) if info_correct else 0)
        
        if all == 0:
            info_aver.append(0)
            title_aver.append(0)
            halluc.append(0)
        else:
            info_aver.append(sum(info_correct)/len(info_correct) if info_correct else 0)
            title_aver.append(sum(correct_title)/len(correct_title) if correct_title else 0)
            halluc.append(1-sum(info_correct)/len(info_correct) if info_correct else 0)
    title_score = average(title_aver)  #题目的正确率
    info_score = average(info_aver)    #
    halluc_score = average(halluc)
    
    # print("title_score", title_score)
    # print("info_score", info_score)
    # print("info_score%", info_score/7)
    # print("halluc_score", halluc_score)
    # exit()
    return title_score, info_score, halluc_score
    
            
   
    

    

    # exit()
def count_json_files(directory):
    json_file_count = 0
    # 遍历文件夹及其所有子文件夹
    for root, dirs, files in os.walk(directory):
        # 统计所有以 .json 结尾的文件
        json_file_count += len([file for file in files if file.endswith('.json')])
        json_files = [file for file in files if file.endswith('.json')]
        # 打印当前文件夹及其中的 .json 文件数量
        # print(f"Folder: {root}, JSON files: {len(json_files)}")
    return json_file_count

def count_json_files_in_subfolders(directory):
    # 遍历第一层子文件夹
    folder_json_counts = []

    for entry in os.scandir(directory):
        if entry.is_dir():  # 只处理子文件夹
            json_file_count = 0
            # 遍历子文件夹中的文件和子文件夹
            for root, dirs, files in os.walk(entry.path):
                json_file_count += len([file for file in files if file.endswith('.json')])
            # 打印每个第一层子文件夹中的 .json 文件数量
            if json_file_count == 0:
                continue
            folder_json_counts.append((entry.name, json_file_count))

            print(f"Folder: {entry.name}, JSON files: {json_file_count}")
    return folder_json_counts


# def count_json_files_in_subfolders(directory):
#     folder_json_counts = []
    
#     # 获取第一层子文件夹
#     for entry in os.scandir(directory):
#         if entry.is_dir():
#             subfolder = entry.path
#             json_file_count = sum(1 for file in os.listdir(subfolder) if file.endswith('.json'))
#             folder_json_counts.append((entry.name, json_file_count))
    
    # return folder_json_counts

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Journal Name', 'Title acc', "Overall hallu score"])
        writer.writerows(data)


if __name__ == "__main__":
    # llm = "claude"
    # llm = "gpt-4o-new"
    # llm = "meta-llama_Llama-3.2-3B-Instruct"
    # llm = "Qwen_Qwen2.5-72B-Instruct"
    llm = "deepseek"
    task = "t3"
    # llm = "llama"
    topics = [f.name for f in os.scandir("./clean/data") if f.is_dir()]

    # topics = ["micro","neuro","nucl","nutr","orgpsych","pathmechdis","pharmtox","physchem","physiol","phyto","arplant","polisci","psych","publhealth","resource","soc","statistics","virology","vision"]
    # topics = ["micro"]

    # topics = [f.name for f in os.scandir("./llm3/gpt-3.5/data") if f.is_dir()]
    # print(len(topics))

    # 统计信息
    # directory = "llm3/meta-llama_Meta-Llama-3.1-8B-Instruct"  # 替换为你要检查的文件夹路径
    # json_files = count_json_files(directory)
    # print(f"Total JSON files: {json_files}")
    # count_data = count_json_files_in_subfolders(directory)
    # save_to_csv(count_data, './count_paper.csv')

    # exit()
    # # '''journal name'''
    # print(topics)
    # exit()
    for t in topics[0:52]:
        # print(t)
        # exit()
        if t in ["arcompsci"]:
            continue
        
        journal_llm_path = './' + task + '/'+ llm + '/data/' + t
        # print(journal_llm_path)
        # exit()
        subdirectories = [d for d in os.listdir(journal_llm_path) if os.path.isdir(os.path.join(journal_llm_path, d))]

        if not os.path.exists(os.path.join("./overlap_results_" + task, journal_llm_path)):
            os.makedirs(os.path.join("./overlap_results_" + task, journal_llm_path))
        #保存分数
        f = open(os.path.join("./overlap_results_" + task, journal_llm_path, t + ".txt"),'w',encoding='utf-8')
        # journal_overlap_scores = []
        journal_title_score, journal_info_score, journal_overlap_score = [], [], []

        for subd in subdirectories:
            # if subd == "Volume 13 (2020)":
                json_files = find_json_files(os.path.join(journal_llm_path, subd))
                # if not os.path.exists(os.path.join("./clean", journal_llm_path, subd)):
                #         os.makedirs(os.path.join("./clean", journal_llm_path, subd))
                
                '''clean data generated form LLM as json format'''
                # clean_references(json_files) #clean the generated results
                # print("finished LLM data cleaning!")

                json_files = find_json_files(os.path.join(journal_llm_path, subd))
                title_score, info_score, overall_score = compute_overlap(json_files, llm, task)
                f.write("############\n")
                f.write(subd + '\n')
                f.write("title_score:\t" + str(title_score) +'\n')
                f.write("info_score:\t" + str(info_score) +'\n')
                f.write("info_score%:\t" + str(info_score/7) +'\n')
                f.write("journal_overlap_score:\t" + str(overall_score) +'\n')
                print("#########################" + t + " " + subd + "#########################")
                print("title_score", title_score)
                print("info_score", info_score)
                print("info_score%", info_score/7)
                print("journal_overlap_score", overall_score)
                journal_title_score.append(title_score)
                journal_info_score.append(info_score)
                journal_overlap_score.append(overall_score)
                # exit()
                
        journal_title_score_ = sum(journal_title_score)/len(journal_title_score)
        journal_info_score_ = sum(journal_info_score)/len(journal_info_score) 
        journal_overlap_score_ = sum(journal_overlap_score)/len(journal_overlap_score)  
        print("#########################" + t + "#########################")
        print("journal_title_score", journal_title_score_)
        print("journal_info_score", journal_info_score_)
        print("journal_info_score%", journal_info_score_/7)
        print("journal_overlap_score", journal_overlap_score_)
        f.write("############\n")
        f.write(t + '\n')
        f.write("journal_title_score:\t" + str(journal_title_score_) +'\n')
        f.write("journal_info_score:\t" + str(journal_info_score_) +'\n')
        f.write("journal_info_score%:\t" + str(journal_info_score_/7) +'\n')
        f.write("journal_overlap_score:\t" + str(journal_overlap_score_) +'\n')
        # exit()
        

            