import sys
sys.path.insert(0, '../transformers/src')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from torch.nn import functional as F
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--detector', type=str, required=True, help='llama2-7b or llama2-13b')
parser.add_argument('--generator', type=str, required=True, help='model field in response.jsonl')
args = parser.parse_args()

response_path = "../dataset/dataset/response.jsonl"
response = []
with open(response_path, 'r') as f:
    for line in f:
        response.append(json.loads(line))

source_info_path = "../dataset/dataset/source_info.jsonl"
source_info_dict = {}
with open(source_info_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        source_info_dict[data['source_id']] = data

if args.detector == "llama2-7b":
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    topk_head_path = "./log/test_llama2_7B/topk_heads.json"
    start, number = 0, 32
elif args.detector == "llama2-13b":
    model_name = "meta-llama/Llama-2-13b-chat-hf"
    topk_head_path = "./log/test_llama2_13B/topk_heads.json"
    start, number = 0, 40
else:
    raise ValueError(f"Unknown detector: {args.detector}")

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if args.detector == "llama2-13b":
    tokenizer_for_temp = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
else:
    tokenizer_for_temp = tokenizer
device = "cuda"

with open(topk_head_path, 'r') as f:
    copy_heads = json.load(f)


def calculate_dist(sep_vocabulary_dist, sep_attention_dist):
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)
    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer)
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(-1)
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').mean(-1)
    js_divs = 0.5 * (kl1 + kl2)
    return js_divs.cpu().item() * 10e5


def is_hallucination_token(token_id, hallucination_spans):
    for span in hallucination_spans:
        if token_id >= span[0] and token_id <= span[1]:
            return True
    return False


def calculate_hallucination_spans(labels, text, response_rag, tokenizer, prefix_len):
    spans = []
    for item in labels:
        start_text = text + response_rag[:item['start']]
        end_text = text + response_rag[:item['end']]
        s = tokenizer(start_text, return_tensors="pt").input_ids.shape[-1]
        e = tokenizer(end_text, return_tensors="pt").input_ids.shape[-1]
        spans.append([s, e])
    return spans


select_response = []
for i in tqdm(range(len(response))):
    if response[i]['model'] != args.generator or response[i]['split'] != 'test':
        continue

    response_rag = response[i]['response']
    source_id = response[i]['source_id']
    prompt = source_info_dict[source_id]['prompt']
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt[:12000]}
    ]
    text = tokenizer_for_temp.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_text = text + response_rag
    input_ids = tokenizer([input_text], return_tensors="pt").input_ids.to(device)
    prefix_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)

    if "labels" in response[i]:
        hallucination_spans = calculate_hallucination_spans(
            response[i]['labels'], text, response_rag, tokenizer, prefix_ids.shape[-1]
        )
    else:
        hallucination_spans = []

    with torch.no_grad():
        logits_dict, outputs = model(
            input_ids=input_ids, return_dict=True,
            output_attentions=True, output_hidden_states=True,
            knowledge_layers=list(range(start, number))
        )
    logits_dict = {k: [v[0].to(device), v[1].to(device)] for k, v in logits_dict.items()}
    hidden_states = outputs["hidden_states"]
    last_hidden_states = hidden_states[-1][0, :, :]

    attentions_list = []
    for lid in range(len(outputs.attentions)):
        for hid in range(outputs.attentions[lid].shape[1]):
            if [lid, hid] not in copy_heads:
                continue
            attentions_list.append(outputs.attentions[lid][:, hid, :, :])

    external_similarity = []
    parameter_knowledge_difference = []
    hallucination_label = []

    for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
        pointer_probs_list = torch.cat([a[:, seq_i, :prefix_ids.shape[-1]] for a in attentions_list], dim=0)
        top_k = int(pointer_probs_list.shape[-1] * 0.1)
        sorted_indices = torch.argsort(pointer_probs_list, dim=1, descending=True)
        top_k_indices = sorted_indices[:, :top_k]
        flattened = top_k_indices.flatten()
        selected = last_hidden_states[flattened]
        top_k_hidden = selected.view(top_k_indices.shape[0], top_k_indices.shape[1], -1)
        attend_mean = torch.mean(top_k_hidden, dim=1)
        current = last_hidden_states[seq_i, :].unsqueeze(0).expand(attend_mean.shape)
        cos_sim = F.cosine_similarity(attend_mean.to(device), current.to(device), dim=1)

        hallucination_label.append(1 if is_hallucination_token(seq_i, hallucination_spans) else 0)
        external_similarity.append(cos_sim.cpu().tolist())
        parameter_knowledge_difference.append(
            [calculate_dist(v[0][0, seq_i, :], v[1][0, seq_i, :]) for v in logits_dict.values()]
        )
        torch.cuda.empty_cache()

    response[i]["external_similarity"] = external_similarity
    response[i]["parameter_knowledge_difference"] = parameter_knowledge_difference
    response[i]["hallucination_label"] = hallucination_label
    select_response.append(response[i])

safe_gen = args.generator.replace("/", "_")
save_path = f"./log/cross_detect_{args.detector}_gen_{safe_gen}.json"
with open(save_path, "w") as f:
    json.dump(select_response, f, ensure_ascii=False)
print(f"Saved {len(select_response)} responses to {save_path}")
