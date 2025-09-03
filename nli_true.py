''''compute the true nli score'''
import argparse
import collections
import json,os
import re,csv
import string
import torch
import copy

from nltk import sent_tokenize
import numpy as np
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
import sys
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import transformers

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)

# from utils import normalize_answer, get_max_memory, remove_citations

# QA_MODEL="gaotianyu1350/roberta-large-squad"
AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"
# /content/drive/MyDrive/bartscore/true-main/offload

global autoais_model, autoais_tokenizer
# autoais_model, autoais_tokenizer = None, None
# offload_folder="offload"

autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, device_map="auto", offload_folder="offload")
autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
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


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """

    input_text = "premise: {} hypothesis: {}".format(passage, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    # print("begin~")

    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference



if __name__ == "__main__":
    llm = "gpt-5"
    # llm = "claude"
    # llm = "Qwen_Qwen2.5-72B-Instruct"
    # llm = "meta-llama_Llama-3.2-3B-Instruct"
    topics = [f.name for f in os.scandir("./t2/" + llm + '/data/') if f.is_dir()]
    # topics = [f.name for f in os.scandir("./clean/data/") if f.is_dir()]
    model_nli = []
    csv_data = []
    for t in topics:
        # print(t)
        # exit()
        if t=="arcompsci":
            continue

        journal_llm_path = './t2/'+ llm + '/data/' + t

        subdirectories = [d for d in os.listdir(journal_llm_path) if os.path.isdir(os.path.join(journal_llm_path, d))]
        journal_title_score, journal_info_score, journal_halluc_score = [], [], []
        for subd in subdirectories:
            # if subd == "Volume 13 (2020)":
            if "2025" in subd:
                json_files = find_json_files(os.path.join(journal_llm_path, subd))
                save_path = os.path.join(journal_llm_path, subd).replace("/t2/", "/t2_truescore/")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for file in json_files:
                    print("#########################################" + file + "##############################################" )
                    save_path = file.replace("/t2/", "/t2_truescore/")
                    generate = read_json_file(file)
                    if isinstance(generate,str):
                        generate = json.loads(generate)
                    abstract_file = file.replace("./t2/" + llm,"./clean")
                    context = read_json_file(abstract_file)
                    abstract1 = context["context"]["ABSTRACT"]
                    score = _run_nli_autoais(abstract1, generate["Abstract"])
                    #   print(score)
                    output = {"grounding": abstract1, "generated_text":generate["Abstract"], "label": score}
                    with open(os.path.join(save_path), 'w') as f:
                        json.dump(output, f, ensure_ascii=False)

    # csv_file_path = "./" + llm + "_nli_score.csv"
    # with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    #   writer = csv.DictWriter(file, fieldnames=["grounding", "generated_text", "label"])
    #   writer.writeheader()  # 写入表头
    #   writer.writerows(csv_data)  # 写入数据
    # print(f"CSV 文件已成功保存至 {csv_file_path}")

