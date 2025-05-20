'''
collect data for three tasks
'''
import os, re, random

import json
import requests

from huggingface_hub import login
from huggingface_hub import InferenceClient
import google.generativeai as genai
import openai
import anthropic
from dashscope import Generation
from openai import OpenAI
os.getenv("OPENAI_API_KEY")
f = open('./prompts/prompts1.txt', 'r', encoding='utf-8')
system_prompt_1 = f.read().strip()

f = open('./prompts/prompts2.txt', 'r', encoding='utf-8')
system_prompt_2 = f.read().strip()

f = open('./prompts/prompts3.txt', 'r', encoding='utf-8')
system_prompt_3 = f.read().strip()

f = open('./prompts/clean.txt', 'r', encoding='utf-8')
clean_prompt = f.read().strip()


user_prompt = ""



def read_json_file(filepath):
    """
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def find_json_files(directory):
    """
    """
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files




def qw_api(system_prompt, user_prompt):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
        ]
    response = Generation.call(
        api_key=os.getenv("Qwen_KEY"), 
        model="qwen2.5-72b-instruct",   # model list：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        temperature=0,
        max_tokens=3700,
        result_format="message"
    )

    if response.status_code == 200:
        result = response.output.choices[0].message.content
        # print(result)
        return result
    else:
        print(f"HTTP：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
def huggingface_api_format(system_prompt, user_prompt, model_id="meta-llama/Meta-Llama-3.2-3B-Instruct"):
    

    messages = [
        { "role": "system", "content": system_prompt},
        { "role": "user", "content": user_prompt}
    ]
    response_format = {
        "type": "json",
        "value": {
            "properties": {
                
                "Literature Review": {
                    "type": "string"
                },

                "References": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "authors": {"type": "string"},
                            "journal": {"type": "string"},
                            "year": {"type": "string"},
                            "volumes": {"type": "string"},
                            "first page": {"type": "string"},
                            "last page": {"type": "string"},
                            "DOI": {"type": "string"}
                        },
                        "required": [
                            "title",
                            "authors",
                            "journal",
                            "year",
                            "volumes",
                            "first page",
                            "last page",
                            "DOI"
                        ]
                    }
                }
            },
            "required": ["Literature Review", "References"]
        }
    }


    client = InferenceClient(api_key=os.getenv("Huggingface_KEY"))
    try:
        completion = client.chat.completions.create(
            model= model_id, 
            messages=messages,
            # response_format=response_format,
            temperature=0,
            max_tokens=3000,
            # top_p=0.7
        )
    except Exception as e:
        print(f"API调用失败，错误：{e}")

    # print(completion.choices[0].message.delta.content)
    content = completion.choices[0].message.content.replace('\n','')

    
    return content

def huggingface_api(system_prompt, user_prompt, model_id="meta-llama/Meta-Llama-3.2-8B-Instruct"):
    

    messages = [
        { "role": "system", "content": system_prompt},
        { "role": "user", "content": user_prompt}
    ]
   
    client = InferenceClient(api_key=os.getenv("Huggingface_KEY"))
    try:
        completion = client.chat.completions.create(
            model= model_id, 
            messages=messages,
            # response_format=response_format,
            temperature=0,
            max_tokens=3500,
            # top_p=0.7
        )
    except Exception as e:
        print(f"API调用失败，错误：{e}")
        
    content = completion.choices[0].message.content.replace('\n','')

    
    return content



def gpt_4o_new(system_prompt, user_prompt):
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

def gpt_4o_old(system_prompt, user_prompt):
    ''' 调用gpt3.5 0.006$/1K token'''
    completion = openai.ChatCompletion.create(
        model="gpt-4o-2024-05-13", 
        messages=[{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    max_tokens=3700)  
    result = completion["choices"][0][ "message"]["content"]
    return result

def gpt(system_prompt, user_prompt):

    ''' 调用gpt3.5 0.006$/1K token'''
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        response_format={
        'type': 'json_object'
    }
    )

    result = response.choices[0].message.content
    return result

def claude(system_prompt, user_prompt):

    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=3700,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user","content": user_prompt},
                {"role": "assistant", "content": "{"}
                ]
                )

    text = [block.text for block in message.content]
    return text[0]

def deepseek(system_prompt, user_prompt):
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        response_format={
        'type': 'json_object'
    }
    )

    return response.choices[0].message.content

def find_file(directory = "./clean/data/anchem/", model_id="gpt-4o", task="refer"):
    if task == "refer":
        task = "t1"
    elif task == "abstract":
        task="t2"
    elif task == "liter":
        task ="t3"
    json_files = find_json_files(directory)
    if not os.path.exists(os.path.join(directory.replace("clean", task + "/" + model_id.replace('/', '_')))):
        os.makedirs(os.path.join(directory.replace("clean", task + "/" + model_id.replace('/', '_'))))
    
    for json_file in json_files:
        # try:
            # page += 1
            # if json_file in ["./clean/data/pharmtox/Volume 63 (2023)/10.1146_annurev-pharmtox-051921-083709_content.json","./clean/data/animal/Volume 11 (2023)/10.1146_annurev-animal-081122-070236_content.json","./clean/data/biochem/Volume 92 (2023)/10.1146_annurev-biochem-032620-104401_content.json","./clean/data/nucl/Volume 73 (2023)/10.1146_annurev-nucl-111422-040200_content.json"]:#,"./clean/data/pharmtox/Volume 63 (2023)/10.1146_annurev-pharmtox-051921-083709_content.json","./clean/data/animal/Volume 11 (2023)/10.1146_annurev-animal-081122-070236_content.json","./clean/data/food/Volume 14 (2023)/10.1146_annurev-food-060721-012235_content.json","./clean/data/food/Volume 14 (2023)/10.1146_annurev-food-060721-015516_content.json","./clean/data/physchem/Volume 74 (2023)/10.1146_annurev-physchem-082720-032137_content.json","./clean/data/physchem/Volume 74 (2023)/10.1146_annurev-physchem-062322-041503_content.json"]:
                # continue
            print("#######################" + json_file + "#############################")
            data = read_json_file(json_file)
            # print(json_file)
            ky = data["keywords"].strip()
            title = data["title"][0].strip()
            abstract_string = data["context"]["ABSTRACT"].strip()
            abstract = data["context"]["ABSTRACT"].strip().split()
            if ky != "" and title != "" and abstract != "":
                if task == "t1" or task == "t2":
                    user_prompt = "Title: " + title + '\n' + "Keywords: " + ky
                elif task == "t3" :
                    user_prompt = "Title: " + title +'\n' + "Abstract: " + abstract_string + "\n" + "Keywords: " + ky 
            else:
                continue
            if task == "t1":
                system_prompt = system_prompt_1
            elif task == "t2":
                system_prompt = system_prompt_2.replace("XX", str(len(abstract)))
            elif task == "t3":
                system_prompt = system_prompt_3

            save_path = json_file.replace("clean", task + "/" + model_id.replace('/', '_'))
            current_dictionary = '/'.join(save_path.split('/')[0:-1])
            # exit()
            # print(current_dictionary )
            if not os.path.exists(current_dictionary):
                os.makedirs(current_dictionary)

            if not os.path.exists(save_path):
                # result = huggingface_api(system_prompt, user_prompt, model_id=model_id)
                # result = geni(system_prompt, user_prompt)
                if model_id == "gpt-4o-new":
                    result = gpt_4o_new(system_prompt, user_prompt)
                    if task == "t3":
                        try:
                            # print(result)
                            result = result.replace('\n','')
                            result = json.loads(result)
                        except:
                            print(type(result))
                            result = gpt(clean_prompt,result)
                            result = result.replace('\n','')
                            print("cleaning results:",result)
                            result = json.loads(result)
                    else:
                        result = json.loads(result)
                    # result = json.loads(result)

                elif model_id == "gpt-3.5":
                    result = gpt(system_prompt, user_prompt)
                    result = json.loads(result)

                elif model_id == "claude":
                    result = claude(system_prompt, user_prompt)
                    result = "{" + result.replace("\n","").replace("    "," ")
                    if task == "t3":
                        try:
                            # print(result)
                            result = result.replace('\n','')
                            result = json.loads(result)
                        except:
                            # print(type(result))
                            result = gpt(clean_prompt,result)
                            result = result.replace('\n','')
                            # print("cleaning results:",result)
                            result = json.loads(result)
                    else:
                        result = json.loads(result)
                elif model_id == "deepseek":
                    result = deepseek(system_prompt, user_prompt)
                    # print(system_prompt, user_prompt)
                    result = result.replace('\n','')
                    # if result.strip() == "":
                    #     continue
                    if task == "t3":
                        try:
                            # print(result)
                            result = json.loads(result)
                        except:
                            print(result)
                            exit()
                            result = gpt(clean_prompt,result)
                            result = result.replace('\n','')
                            print("cleaning results:",result)
                            result = json.loads(result)
                    elif task == "t2":
                        try:
                            result = json.loads(result)
                        except:
                            print(result)
                            # exit()
                            result = gpt(clean_prompt,result)
                            result = result.replace('\n','')
                            print("cleaning results:",result)
                            result = json.loads(result)
                    else:
                        result = json.loads(result)
                    
                elif model_id == "Qwen/Qwen2.5-72B-Instruct":
                    result = qw_api(system_prompt, user_prompt)
                    # result = huggingface_api(system_prompt, user_prompt, model_id)
                    if task == "t2":
                        result = result.replace('{"Abstract": "',"").replace('"}','').replace('"',"'")
                        result = '{"Abstract": "' + result + '"}'
                        result = json.loads(result)
                    elif task == "t3":
                        try:
                            # print(result)
                            result = result.replace('```json','').replace('```','')
                            result = json.loads(result)
                        except:
                            result = gpt(clean_prompt,result)
                            result = result.replace('\n','')
                            print("cleaning results:",result)
                            result = json.loads(result)
                    elif task == "t1":
                        try:
                            result = json.loads(result)
                        except:
                            result = qw_api(system_prompt, user_prompt)
                            result = json.loads(result)
                else:
                    result = huggingface_api(system_prompt, user_prompt, model_id)
                    if task == "t2":
                        result = result.replace('{"Abstract": "',"").replace('"}','').replace('"',"'")
                        result = '{"Abstract": "' + result + '"}'
                        result = json.loads(result)
                    elif task == "t3":
                        try:
                            result = result.replace('```json','').replace('```','')
                            result = json.loads(result)
                        except:
                            result = gpt(clean_prompt,result)
                            # result = huggingface_api(clean_prompt,result,model_id)
                            result = result.replace('\n','')
                            print("cleaning results:",result)
                            result = json.loads(result)
                    elif task == "t1":
                        try:
                            result = json.loads(result)
                        except:
                            result = huggingface_api_format(system_prompt, user_prompt, model_id)
                            print(result)
                            result = json.loads(result)
                with open(save_path, 'w', encoding='utf-8') as f4:
                    json.dump(result, f4, ensure_ascii=False)

if __name__ == "__main__":

    subfolders = [f.name for f in os.scandir("./clean/data/") if f.is_dir()]
    # print(subfolders)
    # exit()
    # for x in ["micro","neuro","nucl","nutr","orgpsych","pathmechdis","pharmtox","physchem","physiol","phyto","arplant","polisci","psych","publhealth","resource","soc","statistics","virology","vision"]:
    # for x in ["anthro","anchem","animal","astro","biochem","biodatasci","bioeng","biophys","cellbio","control","devpsych","ecolsys","economics","energy","financial","fluid","genom","immunol","med"]:
    abstract_empty = 0

    for x in subfolders:
        sub_folders = [f.name for f in os.scandir("./clean/data/" + x +"/") if f.is_dir()]
        for sub in sub_folders:
            if "2023" in sub:
                find_file(directory = "./clean/data/" + x + "/" + sub + "/", model_id="deepseek", task="refer") #Qwen/Qwen2.5-72B-Instruct   mistralai/Mistral-Nemo-Instruct-2407
                #task="refer","abstract","liter"
                
