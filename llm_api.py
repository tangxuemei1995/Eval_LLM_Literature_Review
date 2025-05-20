import os 
# import litellm
# from litellm import completion 
import json
import requests
# [OPTIONAL] set env var
# os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key" 
from huggingface_hub import login
from huggingface_hub import InferenceClient
import google.generativeai as genai
import openai
openai.api_key = ""
# set_openai_key("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")

# login(token="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")

# litellm.set_verbose=True

f = open('./prompt3-gpt.txt', 'r', encoding='utf-8')
# f.read()
system_prompt = f.read().strip()
# system_prompt =  'Act as an experienced researcher, please write a 600-word "Literature Review" based on the following "SUMMARY" and cite 8 relevant articles. Please output the 8 referenced articles in APA format and include the following information in each article: \n \
#     1. title of the article.\n \
#     2. authors.\n \
#     3. journal name.\n \
#     4. year of publication.\n \
#     5. number of volumes.\n \
#     6. Page number of the first page of the article in the journal. \n \
#     7. Page number of the last page of the article in the journal. \n \
#     The output format is json: \n \
#     {\
#     "Literature Review":"", \
#     "References": "{"title":"", \
#                     "authors":"",\
#                     "journal":"", \
#                     "year":"", \
#                     "volumes":"", \
#                     "first page":"", \
#                     "last page":""}" \
#     } \n \
#     Please answer the json directly, without introductory phrases.'


# system_prompt = 'As an experienced researcher If you have to write a "Literature Review" based on the "Abstract" below.\
#                 please Give 16 different and citable literature related to the abstract. Please output the 16 referenced articles in APA format and include the following information in each article: \n \
#     1. title of the article.\n \
#     2. authors.\n \
#     3. journal name.\n \
#     4. year of publication.\n \
#     5. number of volumes.\n \
#     6. Page number of the first page of the article in the journal. \n \
#     7. Page number of the last page of the article in the journal. \n \
#     The output format is json: \n \
#     {\
#     "References": [{"title":"", \
#                     "authors":"",\
#                     "journal":"", \
#                     "year":"", \
#                     "volumes":"", \
#                     "first page":"", \
#                     "last page":""}]\
#     } \n \
#     Please answer the json directly, without introductory phrases.'

user_prompt = ""



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



# def huggingface_api(system_prompt, user_prompt, model_id="google/gemma-2-9b-it"):
#     response = completion(
#         model="huggingface/" + model_id,
#         messages=[{"role": "user", "content": system_prompt},
#                   {"role": "user", "content": user_prompt}],
#         temperature=0.0,
#         max_tokens=3700,
#         details=False,
#         api_key="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",
#         api_base="https://api-inference.huggingface.co/models/" +  model_id)
#     # print(response['choices'][0]['message']['content'])
#     # exit()
#     return response['choices'][0]['message']['content']


def huggingface_api(system_prompt, user_prompt, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    
    API_URL = "https://api-inference.huggingface.co/models/" + model_id
    headers = {"Authorization": f"Bearer hf_key"}
    data = {"inputs": system_prompt + user_prompt}
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()[0]["generated_text"]
    # data = query({"inputs": "Can you please let us know more details about your"})
    # print(data)
    # return 

def geni(system_prompt, user_prompt):
    genai.configure(api_key=os.getenv('GOOGLE_AI_API_KEY'))
# AIzaSyCvCCwkGigepHzZaGcajejx65z670tOsv4
    generation_config = {
        "temperature": 0,
        "top_p": 0.95, # cannot change
        "top_k": 0,
        "max_output_tokens": 250,
        # "response_mime_type": "application/json",
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-exp-0827",
                        generation_config=generation_config,
                        system_instruction=system_prompt,
                        safety_settings=safety_settings)
    convo = model.start_chat(history=[])
    convo.send_message(user_prompt)
    # print(convo.last)
    result = convo.last.text
    print(result)
    return result

# def huggingface_api(system_prompt, user_prompt):
#     client = InferenceClient(
#         "meta-llama/Meta-Llama-3.1-8B-Instruct",
#         token="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",
#     )

#     for message in client.chat_completion(
#         messages=[{"role": "system", "content": system_prompt},
#                   {"role": "user", "content": user_prompt}],
#         max_tokens=3700,
#         stream=True,
#     ):
#         print(message.choices[0].delta.content, end="")
#     exit()


# def gpt(system_prompt, user_prompt):
#     os.environ["OPENAI_API_KEY"] = "sk-None-tanhMyavhUtpX2G1kmPuT3BlbkFJGEhM5jmyGyhrTd3LdHDI"

#     response = completion(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}],
#                 temperature=0.0,
#     max_tokens=3700)  
#     return response['choices'][0]['message']['content']
# os.environ["OPENAI_API_KEY"] = "sk-None-tanhMyavhUtpX2G1kmPuT3BlbkFJGEhM5jmyGyhrTd3LdHDI"
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

def find_file(directory = "./clean/data/anchem/", model_id="gpt-3.5"):
    json_files = find_json_files(directory)
    if not os.path.exists(os.path.join(directory.replace("clean", "llm3/" + model_id.replace('/', '_')))):
        os.makedirs(os.path.join(directory.replace("clean", "llm3/" + model_id.replace('/', '_'))))
    for json_file in json_files:
        # print(json_file)
        # exit()
        try:
            # page += 1
            print("#######################" + json_file + "#############################")
            data = read_json_file(json_file)
            # print(json_file)
            
            user_prompt = "ABSTRACT:" + data["context"]["ABSTRACT"]
             
            # system_prompt = "who you are?"
            # user_prompt = ""
            # result = gpt(system_prompt, user_prompt)
            save_path = json_file.replace("clean", "llm3/" + model_id.replace('/', '_'))
            current_dictionary = '/'.join(save_path.split('/')[0:-1])
            # exit()
            # print(current_dictionary )
            if not os.path.exists(current_dictionary):
                os.makedirs(current_dictionary)

            if not os.path.exists(save_path):
                # result = huggingface_api(system_prompt, user_prompt, model_id=model_id)
                # result = geni(system_prompt, user_prompt)

                result = gpt(system_prompt, user_prompt)
                result = json.loads(result)


                # save_path = json_file.replace("clean", "llm3/llama3"

                with open(save_path, 'w', encoding='utf-8') as f4:
                    json.dump(result, f4, ensure_ascii=False)

                # print(result)

        except (json.JSONDecodeError, OSError) as e:
            print(f"Error reading {json_file}: {e}")
    
if __name__ == "__main__":
    subfolders = [f.name for f in os.scandir("./clean/data/") if f.is_dir()]
    # print(subfolders)
    # exit()
    # for x in ["micro","neuro","nucl","nutr","orgpsych","pathmechdis","pharmtox","physchem","physiol","phyto","arplant","polisci","psych","publhealth","resource","soc","statistics","virology","vision"]:
    # for x in ["anthro","anchem","animal","astro","biochem","biodatasci","bioeng","biophys","cellbio","control","devpsych","ecolsys","economics","energy","financial","fluid","genom","immunol","med"]:
    abstract_empty = 0
    for x in subfolders:
        find_file(directory = "./clean/data/" + x  +"/")
        # json_files = find_json_files(directory = "./clean/data/" + x +"/")
        # for json_file in json_files:
        # print(json_file)
        # exit()
            # print("#######################" + json_file + "#############################")
    #         data = read_json_file(json_file)
    #         # print(json_file)
    #         if data["context"]["ABSTRACT"] == "":
    #             abstract_empty += 1
    #             continue
    # print(abstract_empty)
