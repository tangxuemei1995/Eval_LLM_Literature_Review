'''Use the first author from LLM-generated references as queries to search Semantic Scholar.'''
import argparse
import re
import sys
import json
import requests
import os
import time
def resolve_author(desc: str):
    req_fields = 'authorId,name,url'

    if re.match('\\d+', desc):  # ID given
        rsp = requests.get(f'https://api.semanticscholar.org/graph/v1/author/{desc}',
                           params={'fields': req_fields})
        rsp.raise_for_status()
        return rsp.json()

    else:  # search
        # rsp = requests.get('https://api.semanticscholar.org/graph/v1/author/search',
                        #    params={'query': desc, 'fields': req_fields})
        try:
            rsp = requests.get('https://api.semanticscholar.org/graph/v1/author/search',
                           params={'query': desc, 'fields': req_fields})
            rsp.raise_for_status()
        except:
            print("wait 5 seconds")
            time.sleep(5)
            rsp = requests.get('https://api.semanticscholar.org/graph/v1/author/search',
                        params={'query': desc, 'fields': req_fields})
            
            rsp.raise_for_status()

        results = rsp.json()
        if results['total'] == 0:  # no results
            print(f'Could not find author "{desc}"')
            return []
        elif results['total'] == 1:  # one unambiguous match
            return results['data']
        else:  # multiple matches
            print(f'Multiple authors matched "{desc}".')
            return results["data"]
            # for author in results['data']:
            #     print(author)
            # print('Re-run with a specific ID.')
            # sys.exit(1)


def get_author_papers(author_id):
    rsp = requests.get(f'https://api.semanticscholar.org/graph/v1/author/{author_id}/papers',
                       params={'fields': 'title,url,year', 'limit': 1000})
    rsp.raise_for_status()
    return rsp.json()['data']


def find_authored_papers(left_author_id, right_author_id):
    left_papers = get_author_papers(left_author_id)
    right_papers = get_author_papers(right_author_id)

    left_paper_ids = set(p['paperId'] for p in left_papers)
    right_paper_ids = set(p['paperId'] for p in right_papers)
    coauthored_paper_ids = left_paper_ids.intersection(right_paper_ids)
    authored_papers = [p for p in left_papers if p['paperId'] in coauthored_paper_ids]
    # most recent first, sort by title within any year
    return sorted(authored_papers, key=lambda p: (-p['year'], p['title']))

def search_paper_for_author(authorid):
    url = "https://api.semanticscholar.org/graph/v1/author/" + authorid + "/papers"
    r = requests.get(url,
    params={'fields': 'referenceCount,citationCount,title,authors,venue,url,year,journal,abstract,publicationDate,publicationTypes,isOpenAccess' },
    )
    return r.json()

def normalize_authors(authors):
    return authors.strip().lower()

def clean_authors(authors):
    paper_authors = []
    # authors = authors.replace(" et al.","").replace("...","")
    if not isinstance(authors, list):
        authors = authors.replace(" et al.","").replace("...","")

        authors = authors.strip().replace(" and ", " ., ").replace("&","., ")
        if "." in authors:
            for a in authors.split(".,"):
                paper_authors.append(normalize_authors(a.strip()))
            if  len(paper_authors)==1 and len(paper_authors[0])> 20:
                paper_authors = paper_authors[0].split(",")

        else:
            for a in authors.split(","):
                paper_authors.append(normalize_authors(a.strip()))
    else:
        paper_authors = authors
    paper_authors_ = paper_authors
    paper_authors = []
    for item in paper_authors_:
         item = item.strip()
         if item != '' and item not in paper_authors:
             paper_authors.append(item)
 
    return paper_authors


def find_author_paper(left_author="",title=""):
    out_filepath = "./semantic_author/" + '_'.join(title.replace('"', '').replace('/', '').replace(':', '').split())
    
    author_paper_results = [] #所有作者的paper结果
    if len(out_filepath)>255:
       out_filepath = out_filepath[0:255]
    # print("out_filepath", out_filepath)
    if not os.path.exists(out_filepath):
        os.makedirs(out_filepath)

    if os.path.exists(os.path.join(out_filepath, "semantic_author_paper.json")):
        with open(os.path.join(out_filepath, "semantic_author_paper.json"), 'r', encoding='utf-8') as f:
            try:
                author_paper_results = json.load(f)
            except:
                author_paper_results = []
   
    if author_paper_results == []:
        authors = clean_authors(left_author)
        if authors != []:
            left_author = authors[0] 
        else:
            author_paper_results =  ["can't find match authors in semantic"]
            with open(os.path.join(out_filepath, "semantic_author_paper.json"), 'w', encoding='utf-8') as f2:
                json.dump(author_paper_results, f2, ensure_ascii=False)

        left_author = resolve_author(left_author)
        paper_id = []
        paper_title = []
        one_author_paper_results = [] #存放当前这一个作者对应的paper
        if left_author != []:
            for la in left_author:
                leftid = la['authorId']
                la['name'] = la['name'].replace("/",'')
                if os.path.exists("./match_author/" + leftid + "+" + la['name'] +'.json'):
                    with open("./match_author/" + leftid + "+" + la['name'] +'.json', 'r', encoding='utf-8') as f:
                        one_author_paper_results = json.load(f)
                # print("----------------------------------")
                # print(one_author_paper_results)
                # print("----------------------------------")
                # exit()
                if one_author_paper_results == []:
                    one_author_paper_results = search_paper_for_author(leftid)
                    
                    # print(one_author_paper_results["data"])
                    if "data" in one_author_paper_results:
                        # for paper_info in authored_papers["data"]:
                        #     paper_id.append(paper_info['paperId'])
                        #     paper_title.append(paper_info['title'])
                        author_paper_results += one_author_paper_results["data"]
                        '''根据作者匹配到的文献'''
                        with open("./match_author/" + leftid + "+" + la['name'] +'.json', "w", encoding='utf-8') as f1:
                            json.dump(one_author_paper_results["data"], f1)

                    else:
                        print("authored_papers", one_author_paper_results)
                        exit()
                else:
                    
                    author_paper_results += one_author_paper_results
        else:
            author_paper_results = ["can't find match authors in semantic"]
        with open(os.path.join(out_filepath, "semantic_author_paper.json"), 'w', encoding='utf-8') as f2:
            json.dump(author_paper_results, f2, ensure_ascii=False)
      # print("title_results", title_results)                           
    # if "offset" in str(author_paper_results):
    #     print("offset", author_paper_results)
    #     exit() 
    '''有出现author_paper_results= ['offset', 'data', 'offset', 'data', 'offset']的情况'''
    author_paper_results = [item for item in  author_paper_results if item not in ['offset', 'data']]
   
    return  author_paper_results 
                
    # else:
    #     left_author = resolve_author(left_author)
    #     right_author = resolve_author(right_author)
    #     if left_author != [] and right_author != []:
    #         for la in left_author:
    #             leftid = la['authorId']
    #             for ra in right_author:
    #                 rightid = ra['authorId']
    #         # print(f"Right author: {right_author['name']} {right_author['url']}")
    #                 authored_papers = find_authored_papers(leftid, rightid)
        
                

if __name__ == '__main__':
    find_author_paper(left_author="XXXX", right_author="")