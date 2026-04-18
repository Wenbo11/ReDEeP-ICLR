import sys
sys.path.insert(0, '../transformers/src')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from torch.nn import functional as F
from tqdm import tqdm
import pdb
import pickle
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
if args.dataset == "ragtruth":
    if args.model_name == "llama3-8b":
        response_path = "../dataset/dataset/response_with_llama3_8b.jsonl"
    else:
        response_path = "../dataset/dataset/response.jsonl"
elif args.dataset == "dolly":
    response_path = "../dataset/dataset/response_dolly.jsonl"

response = []
with open(response_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        response.append(data)
if args.dataset == "ragtruth":
    source_info_path = "../dataset/dataset/source_info.jsonl"
elif args.dataset == "dolly":
    source_info_path = "../dataset/dataset/source_info_dolly.jsonl"
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
    topk_head_path = "./log/test_llama3_8B/topk_heads.json"
else:
    print("model name error")
    exit(-1)

with open(topk_head_path,'r') as f:
    # [(layer, head)...]
    copy_heads = json.load(f)


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
        
    return js_divs.cpu().item()*10e5

def calculate_ma_dist(sep_vocabulary_dist, sep_attention_dist):
    sep_vocabulary_dist = F.softmax(sep_vocabulary_dist, dim=-1)

    dist_diff = sep_vocabulary_dist - sep_attention_dist
    # 取绝对值
    abs_diff = torch.abs(dist_diff)

    # 计算 Manhattan 距离
    manhattan_distance = torch.sum(abs_diff)
    
    return manhattan_distance.cpu().item()

def is_hallucination_token(token_id, hallucination_spans):
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
        messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt[:12000]}
                ]
        text = tokenizer_for_temp.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(text)
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

        start_p, end_p = None, None
        if args.model_name == "llama2-7b":
            start = 0 
            number = 32
        elif args.model_name == "llama3-8b":
            start = 0
            number = 32 
        elif args.model_name == "llama2-13b":
            start = 0
            number = 40
        else:
            print("model name error")

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
        attentions_list = []
        for attentions_layer_id in range(len(outputs.attentions)):
            for head_id in range(outputs.attentions[attentions_layer_id].shape[1]):
                if [attentions_layer_id, head_id] not in copy_heads:
                    continue
                attentions_list.append({"layer_head":(attentions_layer_id, head_id), "attention_score":outputs.attentions[attentions_layer_id][:,head_id,:,:]}) 

        # Step 1: Average the attention across the number of heads
        for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):

            # Step 2: Extract the non-zero values from the last row/column
            # Now we gather the attention scores for the last token of each sequence
            pointer_scores_list = [attention_dict["attention_score"][:, seq_i, :] for attention_dict in attentions_list] # shape: (batch_size, sequence_length)

            # Step 3: Perform a softmax over the modified attention scores
            # pointer_probs = nn.F.softmax(pointer_scores, dim=-1)  # shape: (batch_size, sequence_length)
            if start_p != None and  end_p != None:
                pointer_probs_list =  torch.cat([pointer_scores[:,start_p:end_p] for pointer_scores in pointer_scores_list], dim=0)
            else:
                pointer_probs_list =  torch.cat([pointer_scores[:,:prefix_ids.shape[-1]] for pointer_scores in pointer_scores_list], dim=0)   # shape: (batch_size, prefix_sequence_length) 截取这一步还是只让模型关注文本内容

            # Step 4: select the top attented token
            # Create an extended attention mask that masks out special tokens
            # hyperparameter: token rate

            # pointer_probs_list 是每个位置对应的大小(head_num, seq_len)，last_hidden_states shape (seq_len, hidden_state)是每个位置对应的 value，请取出 top 10% input_ids_cp 的 last_hidden_states，最终输出为(head_num, top10_len, hidden_state)
            # 获取top 10%的索引
            top_k = int(pointer_probs_list.shape[-1] * 0.1)  # 10% of sequence length

            # 获取排序后的索引，按照概率从大到小排序
            sorted_indices = torch.argsort(pointer_probs_list, dim=1, descending=True)

            # 选择前top_k个索引
            top_k_indices = sorted_indices[:, :top_k]

            # 我们需要将 top_k_indices 展平，以便用于索引 last_hidden_states
            flattened_indices = top_k_indices.flatten()  # shape (head_num * k,)
            # 使用展平的索引在 last_hidden_states 中查找相应的 hidden_state
            selected_hidden_states = last_hidden_states[flattened_indices]  # shape (head_num * k, hidden_state)
            # 重新 reshape 成 (head_num, k, hidden_state)
            top_k_hidden_states = selected_hidden_states.view(top_k_indices.shape[0], top_k_indices.shape[1], -1)

            attend_token_hidden_state = torch.mean(top_k_hidden_states, dim=1) # (head_num, hidden_state)

            # Step 5: Calculate the similarity between the last token and the attentioned prefix text
            current_hidden_state = last_hidden_states[seq_i, :] # shape (hidden_state,)

            # 扩展 current_hidden_state 的形状以匹配 pointer_probs_list
            current_hidden_state = current_hidden_state.unsqueeze(0).expand(attend_token_hidden_state.shape)

            # 计算余弦相似度
            cosine_similarity = F.cosine_similarity(attend_token_hidden_state.to(device), current_hidden_state.to(device), dim=1)
            if is_hallucination_token(seq_i, hallucination_spans):
                hallucination_label.append(1)
            else:
                hallucination_label.append(0)
            external_similarity.append(cosine_similarity.cpu().tolist())
            parameter_knowledge_difference.append([calculate_dist(value[0][0,seq_i,:], value[1][0,seq_i,:]) for value in logits_dict.values()])
            torch.cuda.empty_cache()
        response[i]["external_similarity"] = external_similarity
        response[i]["parameter_knowledge_difference"] = parameter_knowledge_difference
        response[i]["hallucination_label"] = hallucination_label


        select_response.append(response[i])
        # if len(select_response)>10:
        #     break

# with open("./data/llama2_7B_response.json", "w") as f:
#     json.dump(select_response, f, indent=4, ensure_ascii=False)

if args.model_name == "llama2-7b":
    if args.dataset == "ragtruth":
        save_path = "./log/test_llama2_7B/llama2_7B_response_v1.json"
    elif args.dataset == "dolly":
        save_path = "./log/test_llama2_7B/llama2_7B_response_v1_dolly.json"
elif args.model_name == "llama2-13b":
    if args.dataset == "ragtruth":
        save_path = "./log/test_llama2_13B/llama2_13B_response_v1.json"
    elif args.dataset == "dolly":
        save_path = "./log/test_llama2_13B/llama2_13B_response_v1_dolly.json"
elif args.model_name == "llama3-8b":
    if args.dataset == "ragtruth":
        save_path = "./log/test_llama3_8B/llama3_8B_response_v1.json"
    elif args.dataset == "dolly":
        save_path = "./log/test_llama3_8B/llama3_8B_response_v1_dolly.json"
else:
    print("model name error")
    exit(-1)

with open(save_path, "w") as f:
    json.dump(select_response, f, ensure_ascii=False)  
