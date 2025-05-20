import requests
import os
import json
from find_paper_by_author import find_author_paper
# Define the API endpoint URL
url = 'https://api.semanticscholar.org/graph/v1/paper/search'

# More specific query parameter
query_params = {'query': 'quantum computing'}

# Directly define the API key (Reminder: Securely handle API keys in production environments)
# api_key = 'your api key goes here'  # Replace with the actual API key

# # Define headers with API key
# headers = {'x-api-key': api_key}

# Send the API request
def semantic_find_title(title):
   response = requests.get("https://api.semanticscholar.org/graph/v1/paper/search/match?query=" + title)
                        # headers=headers)

# Check response status
   flag = 1
   response_data = {}
   if response.status_code == 200:
      response_data = response.json()
      # Process and print the response data as needed
      # print(response_data)

   else:
      print(f"Request failed with status code {response.status_code}: {response.text}")
      print(title + " " + response.text)
      if "Title match not found" in response.text:
         print("\n\n###############Title match not found#################\n" + title + "\n")
         response_data = {"data": response.text}
      else:
         print("\n\n###############this generated paper has a specific error:#################\n\n", response.text)
         response_data = {"data": response.text}
      flag = 0
   return response_data, flag




def search_information(ids):
   r = requests.post(
      'https://api.semanticscholar.org/graph/v1/paper/batch',
      params={'fields': 'referenceCount,citationCount,title,authors,venue,url,year,journal,abstract,publicationDate,publicationTypes,isOpenAccess' },
      json={"ids": ids}
   )
   return r.json()


def semantic_scholar_search(title):
   
   out_filepath = "./semantic/" + '_'.join(title.replace('"', '').replace('/', '').replace(':', '').split())
   # scholar = Scholar(out_filepath)
   # print("#######################" + title + "find it in semantic scholar##############################")   if len(out_filepath)>255:
   out_filepath = out_filepath[0:255]
   # if out_filepath == "./semantic/2015_Eligibility_and_Disqualification_Recommendations_for_Competitive_Athletes_with_Cardiovascular_Abnormalities_Task_Force_3_Hypertrophic_Cardiomyopathy,_Arrhythmogenic_Right_Ventricular_Cardiomyopathy_and_Other_Cardiomyopathies,_and_Myocarditis_A_Scientific_Statement_From_the_American_Heart_Association_and_American_College_of_Cardiology":
   # if len(out_filepath) > 50:
   #    out_filepath = out_filepath[0:50]
   semantic_results, title_results = [] ,{} 
   paper_title_exsit = False 
   if os.path.exists(os.path.join(out_filepath, "semantic.json")):
         with open(os.path.join(out_filepath, "semantic.json"), 'r', encoding='utf-8') as f:
            semantic_results = json.load(f)
         if semantic_results != []:
            if semantic_results =="{'error': 'No valid paper ids given'}":
               semantic_results = []
            if isinstance(semantic_results, dict) :
               # print("semantic_results",semantic_results)
               if "error" in semantic_results.keys():
                  semantic_results = []
               elif "message" in semantic_results.keys():
                  if "Too Many Requests" in semantic_results["message"]:
                     semantic_results = []
            elif isinstance(semantic_results, list):
               paper_title_exsit = True

   if semantic_results == []:
      if not os.path.exists(os.path.join(out_filepath)):
         os.makedirs(out_filepath)
      if os.path.exists(os.path.join(out_filepath, "semantic_title.jsonn")):
         with open(os.path.join(out_filepath, "semantic_title.jsonn"), 'r', encoding='utf-8') as f:
            title_results = json.load(f)
      if title_results == {} or "Too Many Requests" in title_results["data"]:
         title_results, flag = {"data":"Too Many Requests"}, False #说明当前数据需要重新查询
      elif "Title match not found" in title_results["data"]:
         title_results, flag = {"data": "{\"error\":\"Title match not found\"}\n"}, False #说明已经查询过，并且文章不存在
      else:
         title_results, flag = title_results, True

      # print("title_results",title_results)
      if "Too Many Requests" in title_results["data"] or '{"message": "Internal Server Error"}\n' in title_results["data"] or '{"error":"Missing required parameter: \'query\'"}\n' in title_results["data"] or title_results["data"]== '{"message": "Network error communicating with endpoint"}'or title_results["data"] == '{"message": "Endpoint request timed out"}':
         title_results, flag = semantic_find_title(title)
         with open(os.path.join(out_filepath, "semantic_title.jsonn"), 'w', encoding='utf-8') as f:
            json.dump(title_results, f, ensure_ascii=False)
      # print("title_results", title_results)
      if flag != 0:
         '''有对应的合适的题目'''
         if "data" in title_results.keys():
            ids = []
            for paper in title_results["data"]:
               # print("semantic_title_results paper",paper)

               candidate_paper_title = paper["title"]
               candidate_paper_id = paper["paperId"]

               ids.append(candidate_paper_id)
               # print("\n#########################candidate ids#################\n" + ','.join(ids) + "\n")
            paper_information = search_information(ids)
            semantic_results = paper_information 
            if semantic_results == "{'error': 'No valid paper ids given'}":
               semantic_results = {}  
               paper_title_exsit =  False   
            with open(os.path.join(out_filepath, "semantic.json"), 'w', encoding='utf-8') as f:
               json.dump(semantic_results, f, ensure_ascii=False)
            paper_title_exsit =  True
      else:
         paper_title_exsit =  False
   return title_results, semantic_results, paper_title_exsit


if __name__ == "__main__":
   # print(semantic_scholar_search(""))