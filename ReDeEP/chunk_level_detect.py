import sys
sys.path.insert(0, '../transformers/src')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
import json
from torch.nn import functional as F
from tqdm import tqdm
import pdb
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Script for processing data and models.')
parser.add_argument('--model_name', type=str, required=True, help='llama2-7b or llama2-13b or llama3-8b')
parser.add_argument(
    '--dataset', 
    type=str, 
    default="ragtruth", 
    help='ragtruth, dolly'
)

args = parser.parse_args()


bge_model = SentenceTransformer('BAAI/bge-base-en-v1.5').to("cuda:0")
if args.dataset == "ragtruth":
    if args.model_name == "llama3-8b":
        response_path = "../dataset/dataset/response_with_llama3_8b_spans.jsonl"
    else:
        response_path = "../dataset/dataset/response_spans.jsonl"
elif args.dataset == "dolly":
    response_path = "../dataset/dataset/response_dolly_spans.jsonl"
response = []
with open(response_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        response.append(data)
if args.dataset == "ragtruth":
    source_info_path = "../dataset/dataset/source_info_spans.jsonl"
elif args.dataset == "dolly":
    source_info_path = "../dataset/dataset/source_info_dolly_spans.jsonl"
source_info_dict = {}

with open(source_info_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        source_info_dict[data['source_id']] = data


if args.model_name == "llama2-7b":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
elif args.model_name == "llama2-13b":
    model_name = "meta-llama/Llama-2-13b-chat-hf"
elif args.model_name == "llama3-8b":
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
else:
    print("name error")
    exit(-1)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda"

if args.model_name == "llama2-13b":
    tokenizer_for_temp = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
else:
    tokenizer_for_temp = tokenizer


if args.model_name == "llama2-7b":
    topk_head_path = "./log/test_llama2_7B/topk_heads.json"
elif args.model_name == "llama2-13b":
    topk_head_path = "./log/test_llama2_13B/topk_heads.json"
elif args.model_name == "llama3-8b":
    topk_head_path = "./log/test_llama3_8B/topk_heads.json" #"./log/test_llama3_8B/topk_heads.json"
else:
    print("model name error")
    exit(-1)

with open(topk_head_path,'r') as f:
    # [(layer, head)...]
    copy_heads = json.load(f)[:32]


def calculate_dist(sep_vocabulary_dist, sep_attention_dist):
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)  
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)  

    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer) 

    # 4. Calculate log-softmax for the KL divergence
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)  
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1) 

    # 5. Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(-1)  
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').mean(-1)  
    # # Fix bug: https://github.com/Jeryi-Sun/ReDEeP-ICLR/issues/2 but for stable calculation, we maintain the original implementation of JSD.
    # kl1 = F.kl_div(M.log(), softmax_mature.unsqueeze(0),  reduction='none').mean(-1)
    # kl2 = F.kl_div(M.log(), softmax_anchor,  reduction='none').mean(-1)
    js_divs = 0.5 * (kl1 + kl2) 
        
    return js_divs.cpu().item()*10e5 # 这边有个 10e5, 下面没有


def calculate_dist_2d(sep_vocabulary_dist, sep_attention_dist):
    # Calculate softmax
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)  
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)  

    # Calculate the average distribution M
    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer) 

    # Calculate log-softmax for the KL divergence
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)  # Adding epsilon for numerical stability
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)   # Adding epsilon for numerical stability

    # Calculate the KL divergences and then the JS divergences
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').sum(dim=-1)  
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').sum(dim=-1)  
    js_divs = 0.5 * (kl1 + kl2)
    
    scores = js_divs.cpu().tolist()
    
    return sum(scores)

def calculate_ma_dist(sep_vocabulary_dist, sep_attention_dist):
    sep_vocabulary_dist = F.softmax(sep_vocabulary_dist, dim=-1)

    dist_diff = sep_vocabulary_dist - sep_attention_dist
    # 取绝对值
    abs_diff = torch.abs(dist_diff)

    # 计算 Manhattan 距离
    manhattan_distance = torch.sum(abs_diff)
    
    return manhattan_distance.cpu().item()

def add_special_template(prompt):
    messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
    text = tokenizer_for_temp.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text

def is_hallucination_token(token_id, hallucination_spans):
    for span in hallucination_spans:
        if token_id >= span[0] and token_id <= span[1]:
            return True
    return False

def is_hallucination_span(r_span, hallucination_spans):
    for token_id in range(r_span[0], r_span[1]):
        for span in hallucination_spans:
            if token_id >= span[0] and token_id <= span[1]:
                return True
    return False
def calculate_hallucination_spans(response, text, response_rag, tokenizer, prefix_len):
    hallucination_span = []
    if "dolly" in source_info_path:
        return hallucination_span
    for item in response:
        start_id = item['start']
        end_id = item['end']
        start_text = text+response_rag[:start_id]
        end_text = text+response_rag[:end_id]
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]
        end_id = end_text_id.shape[-1]
        hallucination_span.append([start_id, end_id])
    return hallucination_span

def calculate_respond_spans(raw_response_spans, text, response_rag, tokenizer):
    respond_spans = []
    for item in raw_response_spans:
        start_id = item[0]
        end_id = item[1]
        start_text = text+response_rag[:start_id]
        end_text = text+response_rag[:end_id]
        start_text_id = tokenizer(start_text, return_tensors="pt").input_ids
        end_text_id = tokenizer(end_text, return_tensors="pt").input_ids
        start_id = start_text_id.shape[-1]
        end_id = end_text_id.shape[-1]
        respond_spans.append([start_id, end_id])
    return respond_spans


def calculate_prompt_spans(raw_prompt_spans, prompt, tokenizer):
    prompt_spans = []
    for item in raw_prompt_spans:
        start_id = item[0]
        end_id = item[1]
        start_text = prompt[:start_id]
        end_text = prompt[:end_id]
        added_start_text = add_special_template(start_text)
        added_end_text = add_special_template(end_text)
        start_text_id = tokenizer(added_start_text, return_tensors="pt").input_ids.shape[-1] - 4
        end_text_id = tokenizer(added_end_text,return_tensors="pt").input_ids.shape[-1] -4
        prompt_spans.append([start_text_id, end_text_id])
    return prompt_spans

def calculate_sentence_similarity(r_text, p_text):
    part_embedding = bge_model.encode([r_text], normalize_embeddings=True)
    q_embeddings = bge_model.encode([p_text], normalize_embeddings=True)

    # 计算得分
    scores_named = np.matmul(q_embeddings, part_embedding.T).flatten()
    return float(scores_named[0])


select_response = []
if args.model_name == "llama2-7b":
    data_type = "llama-2-7b-chat"
elif args.model_name == "llama2-13b":
    data_type = "llama-2-13b-chat"
elif args.model_name == "llama3-8b":
    data_type =  "llama-3-8b-instruct" 
else:
    print("model name error")
    exit(-1) 


for i in tqdm(range(len(response))):
    if response[i]['model'] == data_type and response[i]["split"] == "test":
        response_rag = response[i]['response']
        source_id = response[i]['source_id']
        temperature = response[i]['temperature']
        prompt =  source_info_dict[source_id]['prompt']
        original_prompt_spans = source_info_dict[source_id]['prompt_spans']
        original_response_spans = response[i]['response_spans']

        text = add_special_template(prompt[:12000])
        input_text = text+response_rag
        print("all_text_len:", len(input_text))
        print("prompt_len", len(prompt))
        print("respond_len", len(response_rag))
        input_ids = tokenizer([input_text], return_tensors="pt").input_ids.to(device)
        prefix_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
        continue_ids = input_ids[0, prefix_ids.shape[-1]:]

        if "labels" in response[i].keys():
            hallucination_spans = calculate_hallucination_spans(response[i]['labels'], text, response_rag, tokenizer, prefix_ids.shape[-1])
        else:
            hallucination_spans = []

        prompt_spans = calculate_prompt_spans(source_info_dict[source_id]['prompt_spans'], prompt, tokenizer)
        respond_spans = calculate_respond_spans(response[i]['response_spans'], text, response_rag, tokenizer)
        if args.model_name == "llama2-7b":
            start = 0 
            number = 32
        elif args.model_name == "llama3-8b":
            start = 0
            number = 16
        elif args.model_name == "llama2-13b":
            start = 8
            number = 40
        else:
            print("model name error")

        start_p, end_p = None, None
        with torch.no_grad():
            logits_dict, outputs = model(
                    input_ids=input_ids, 
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=True,
                    knowledge_layers=list(range(start, number))
                )
        logits_dict = {key: [value[0].to(device), value[1].to(device)] for key, value in logits_dict.items()}

        # skip tokens without hallucination
        hidden_states = outputs["hidden_states"] # tuple ([batch, seq_len, vocab_size], ..., ) 
        last_hidden_states = hidden_states[-1][0, :, :] # [prefix_len, hidden_size]
        
        # todo 修改成 筛选 teacher focusing 的 token 和 model generate token 是否在 top_10内
        # probs = outputs['logits'][range(outputs["logits"].shape[0]), continue_ids].sum().item()
        # # ---------------------------------------------------------------------------------------------------------------
        external_similarity = [] # 这个用来存储生成的 token embedding 和 copy head 关注的 token embedding 的相似度得分
        parameter_knowledge_difference = []
        hallucination_label = []
        # 计算一下输入的 context 里面有没有 hallucination 词，如果有的话 copy 的时候把他们的 pointer weight 调小
        # input: input_ids, corr token vocab distribution
        # output: hallucination score for the input_ids or hallucination mask
        # outputs.attentions is a tuple, taking the last layer's attentions
        span_socre_dict = []
        for r_id, r_span in enumerate(respond_spans):
            layer_head_span = {}
            for attentions_layer_id in range(len(outputs.attentions)):
                for head_id in range(outputs.attentions[attentions_layer_id].shape[1]):
                    if [attentions_layer_id, head_id] in copy_heads:
                        layer_head = (attentions_layer_id, head_id)
                        p_span_score_dict = []
                        for p_span in prompt_spans:
                            attention_score = outputs.attentions[attentions_layer_id][0,head_id,:,:]
                            p_span_score_dict.append([p_span, torch.sum(attention_score[r_span[0]:r_span[1], p_span[0]:p_span[1]]).cpu().item()])
                        # 取出最大的 score 对应的 p_span
                        p_id = max(range(len(p_span_score_dict)), key=lambda i: p_span_score_dict[i][1])
                        prompt_span_text, respond_span_text = prompt[original_prompt_spans[p_id][0]:original_prompt_spans[p_id][1]], response_rag[original_response_spans[r_id][0]:original_response_spans[r_id][1]]
                        # 取出排序后列表的第一个元素的键
                        layer_head_span[str(layer_head)] = calculate_sentence_similarity(prompt_span_text, respond_span_text)

            parameter_knowledge_scores = [calculate_dist_2d(value[0][0,r_span[0]:r_span[1],:], value[1][0,r_span[0]:r_span[1],:]) for value in logits_dict.values()]
            parameter_knowledge_dict = {f"layer_{i}": value for i, value in enumerate(parameter_knowledge_scores)}

            span_socre_dict.append({
                "prompt_attention_score":layer_head_span, # 
                "r_span": r_span,
                "hallucination_label": 1 if is_hallucination_span(r_span, hallucination_spans) else 0,
                "parameter_knowledge_scores": parameter_knowledge_dict
            }) 

        response[i]["scores"] = span_socre_dict
        select_response.append(response[i])

if args.model_name == "llama2-7b":
    if args.dataset == "ragtruth":
        save_path = "./log/test_llama2_7B/llama2_7B_response_chunk.json"
    elif args.dataset == "dolly":
        save_path = "./log/test_llama2_7B/llama2_7B_response_chunk_dolly.json"
elif args.model_name == "llama2-13b":
    if args.dataset == "ragtruth":
        save_path = "./log/test_llama2_13B/llama2_13B_response_chunk.json"
    elif args.dataset == "dolly":
        save_path = "./log/test_llama2_13B/llama2_13B_response_chunk_dolly.json"
elif args.model_name == "llama3-8b":
    if args.dataset == "ragtruth":
        save_path = "./log/test_llama3_8B/llama3_8B_response_chunk.json"
    elif args.dataset == "dolly":
        save_path = "./log/test_llama3_8B/llama3_8B_response_chunk_dolly.json"
else:
    print("model name error")
    exit(-1)

with open(save_path, "w") as f:
    json.dump(select_response, f, ensure_ascii=False)        

    
