import sys
sys.path.insert(0, '../transformers/src')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from torch.nn import functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--detector', type=str, required=True, help='llama2-7b or llama2-13b')
parser.add_argument('--generator', type=str, required=True, help='model field in response_spans.jsonl')
args = parser.parse_args()

bge_model = SentenceTransformer('BAAI/bge-base-en-v1.5').to("cuda:0")

response_path = "../dataset/dataset/response_spans.jsonl"
response = []
with open(response_path, 'r') as f:
    for line in f:
        response.append(json.loads(line))

source_info_path = "../dataset/dataset/source_info_spans.jsonl"
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
    start, number = 8, 40
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
    copy_heads = json.load(f)[:32]


def calculate_dist_2d(sep_vocabulary_dist, sep_attention_dist):
    softmax_mature_layer = F.softmax(sep_vocabulary_dist, dim=-1)
    softmax_anchor_layer = F.softmax(sep_attention_dist, dim=-1)
    M = 0.5 * (softmax_mature_layer + softmax_anchor_layer)
    log_softmax_mature_layer = F.log_softmax(sep_vocabulary_dist, dim=-1)
    log_softmax_anchor_layer = F.log_softmax(sep_attention_dist, dim=-1)
    kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').sum(dim=-1)
    kl2 = F.kl_div(log_softmax_anchor_layer, M, reduction='none').sum(dim=-1)
    js_divs = 0.5 * (kl1 + kl2)
    return sum(js_divs.cpu().tolist())


def add_special_template(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return tokenizer_for_temp.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def is_hallucination_span(r_span, hallucination_spans):
    for token_id in range(r_span[0], r_span[1]):
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


def calculate_respond_spans(raw_response_spans, text, response_rag, tokenizer):
    respond_spans = []
    for item in raw_response_spans:
        start_text = text + response_rag[:item[0]]
        end_text = text + response_rag[:item[1]]
        s = tokenizer(start_text, return_tensors="pt").input_ids.shape[-1]
        e = tokenizer(end_text, return_tensors="pt").input_ids.shape[-1]
        respond_spans.append([s, e])
    return respond_spans


def calculate_prompt_spans(raw_prompt_spans, prompt, tokenizer):
    prompt_spans = []
    for item in raw_prompt_spans:
        added_start = add_special_template(prompt[:item[0]])
        added_end = add_special_template(prompt[:item[1]])
        s = tokenizer(added_start, return_tensors="pt").input_ids.shape[-1] - 4
        e = tokenizer(added_end, return_tensors="pt").input_ids.shape[-1] - 4
        prompt_spans.append([s, e])
    return prompt_spans


def calculate_sentence_similarity(r_text, p_text):
    part_embedding = bge_model.encode([r_text], normalize_embeddings=True)
    q_embeddings = bge_model.encode([p_text], normalize_embeddings=True)
    scores = np.matmul(q_embeddings, part_embedding.T).flatten()
    return float(scores[0])


select_response = []
for i in tqdm(range(len(response))):
    if response[i]['model'] != args.generator or response[i]['split'] != 'test':
        continue

    response_rag = response[i]['response']
    source_id = response[i]['source_id']
    prompt = source_info_dict[source_id]['prompt']
    original_prompt_spans = source_info_dict[source_id]['prompt_spans']
    original_response_spans = response[i]['response_spans']

    text = add_special_template(prompt[:12000])
    input_text = text + response_rag
    input_ids = tokenizer([input_text], return_tensors="pt").input_ids.to(device)
    prefix_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)

    if "labels" in response[i]:
        hallucination_spans = calculate_hallucination_spans(
            response[i]['labels'], text, response_rag, tokenizer, prefix_ids.shape[-1]
        )
    else:
        hallucination_spans = []

    prompt_spans = calculate_prompt_spans(original_prompt_spans, prompt, tokenizer)
    respond_spans = calculate_respond_spans(original_response_spans, text, response_rag, tokenizer)

    with torch.no_grad():
        logits_dict, outputs = model(
            input_ids=input_ids, return_dict=True,
            output_attentions=True, output_hidden_states=True,
            knowledge_layers=list(range(start, number))
        )
    logits_dict = {k: [v[0].to(device), v[1].to(device)] for k, v in logits_dict.items()}

    span_score_dict = []
    for r_id, r_span in enumerate(respond_spans):
        layer_head_span = {}
        for attentions_layer_id in range(len(outputs.attentions)):
            for head_id in range(outputs.attentions[attentions_layer_id].shape[1]):
                if [attentions_layer_id, head_id] in copy_heads:
                    layer_head = (attentions_layer_id, head_id)
                    p_span_score_dict = []
                    for p_span in prompt_spans:
                        attention_score = outputs.attentions[attentions_layer_id][0, head_id, :, :]
                        p_span_score_dict.append([p_span, torch.sum(attention_score[r_span[0]:r_span[1], p_span[0]:p_span[1]]).cpu().item()])
                    p_id = max(range(len(p_span_score_dict)), key=lambda idx: p_span_score_dict[idx][1])
                    prompt_span_text = prompt[original_prompt_spans[p_id][0]:original_prompt_spans[p_id][1]]
                    respond_span_text = response_rag[original_response_spans[r_id][0]:original_response_spans[r_id][1]]
                    layer_head_span[str(layer_head)] = calculate_sentence_similarity(prompt_span_text, respond_span_text)

        parameter_knowledge_scores = [
            calculate_dist_2d(v[0][0, r_span[0]:r_span[1], :], v[1][0, r_span[0]:r_span[1], :])
            for v in logits_dict.values()
        ]
        parameter_knowledge_dict = {f"layer_{idx}": val for idx, val in enumerate(parameter_knowledge_scores)}

        span_score_dict.append({
            "prompt_attention_score": layer_head_span,
            "r_span": r_span,
            "hallucination_label": 1 if is_hallucination_span(r_span, hallucination_spans) else 0,
            "parameter_knowledge_scores": parameter_knowledge_dict
        })

    response[i]["scores"] = span_score_dict
    select_response.append(response[i])

safe_gen = args.generator.replace("/", "_")
save_path = f"./log/cross_chunk_{args.detector}_gen_{safe_gen}.json"
with open(save_path, "w") as f:
    json.dump(select_response, f, ensure_ascii=False)
print(f"Saved {len(select_response)} responses to {save_path}")
