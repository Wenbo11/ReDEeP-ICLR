"""Score chunk-level cross-model detection: all generators combined in one dataframe."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--detector', type=str, required=True)
parser.add_argument('--files', type=str, nargs='+', required=True, help='paths to chunk detection JSONs')
parser.add_argument('--top_n', type=int, required=True)
parser.add_argument('--top_k', type=int, required=True)
parser.add_argument('--alpha', type=float, required=True)
parser.add_argument('--number', type=int, default=32)
args = parser.parse_args()

source_info_path = "../dataset/dataset/source_info.jsonl"
source_info_dict = {}
with open(source_info_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        source_info_dict[data['source_id']] = data

data_dict = {
    "identifier": [],
    "type": [],
    **{f"external_similarity_{k}": [] for k in range(args.number)},
    **{f"parameter_knowledge_difference_{k}": [] for k in range(args.number)},
    "hallucination_label": []
}

global_idx = 0
for fpath in args.files:
    with open(fpath, "r") as f:
        response = json.load(f)
    for i, resp in enumerate(response):
        if resp["split"] != "test":
            continue
        source_id = resp["source_id"]
        rep_type = source_info_dict[source_id]["task_type"]
        for j in range(len(resp["scores"])):
            data_dict["identifier"].append(f"response_{global_idx}_item_{j}")
            data_dict["type"].append(rep_type)
            for k in range(args.number):
                data_dict[f"external_similarity_{k}"].append(
                    list(resp["scores"][j]["prompt_attention_score"].values())[k]
                )
                data_dict[f"parameter_knowledge_difference_{k}"].append(
                    list(resp["scores"][j]["parameter_knowledge_scores"].values())[k]
                )
            data_dict["hallucination_label"].append(resp["scores"][j]["hallucination_label"])
        global_idx += 1

df = pd.DataFrame(data_dict)
print(f"Combined dataframe: {len(df)} spans from {global_idx} responses")
print(f"Hallucination distribution:\n{df['hallucination_label'].value_counts(normalize=True)}")

auc_ext = []
auc_param = []
for k in range(args.number):
    auc_e = roc_auc_score(1 - df['hallucination_label'], df[f'external_similarity_{k}'])
    auc_ext.append((auc_e, f'external_similarity_{k}'))
    auc_p = roc_auc_score(df['hallucination_label'], df[f'parameter_knowledge_difference_{k}'])
    auc_param.append((auc_p, f'parameter_knowledge_difference_{k}'))

top_ext = sorted(auc_ext, reverse=True)[:args.top_n]
top_par = sorted(auc_param, reverse=True)[:args.top_k]

print(f"\nTop-{args.top_n} ext features: {[(f'{a:.4f}', c) for a,c in top_ext]}")
print(f"Top-{args.top_k} param features: {[(f'{a:.4f}', c) for a,c in top_par]}")

df['ext_sum'] = df[[c for _, c in top_ext]].sum(axis=1)
df['param_sum'] = df[[c for _, c in top_par]].sum(axis=1)

scaler = MinMaxScaler()
df['ext_norm'] = scaler.fit_transform(df[['ext_sum']])
df['param_norm'] = scaler.fit_transform(df[['param_sum']])
df['diff'] = df['param_norm'] - args.alpha * df['ext_norm']

df['response_group'] = df['identifier'].str.extract(r'(response_\d+)')
grouped = df.groupby('response_group').agg(
    diff_mean=('diff', 'mean'),
    hallucination_label=('hallucination_label', 'max'),
    resp_type=('type', 'first')
).reset_index()

mn, mx = grouped['diff_mean'].min(), grouped['diff_mean'].max()
grouped['score'] = (grouped['diff_mean'] - mn) / (mx - mn)

auc = roc_auc_score(grouped['hallucination_label'], grouped['score'])
pcc, _ = pearsonr(grouped['hallucination_label'], grouped['score'])
n_total = len(grouped)
n_pos = int(grouped['hallucination_label'].sum())

preds_05 = (grouped['score'] >= 0.5).astype(int)
rec_05 = recall_score(grouped['hallucination_label'], preds_05, zero_division=0)
prec_05 = precision_score(grouped['hallucination_label'], preds_05, zero_division=0)
f1_05 = f1_score(grouped['hallucination_label'], preds_05, zero_division=0)

best_f1, best_t, best_rec, best_prec = 0, 0.5, 0, 0
for t in np.arange(0.01, 1.0, 0.01):
    p = (grouped['score'] >= t).astype(int)
    f = f1_score(grouped['hallucination_label'], p, zero_division=0)
    if f > best_f1:
        best_f1 = f
        best_t = t
        best_rec = recall_score(grouped['hallucination_label'], p, zero_division=0)
        best_prec = precision_score(grouped['hallucination_label'], p, zero_division=0)

print(f"\n=== COMBINED ({args.detector}, {n_total} responses, {n_pos} hallucinated) ===")
print(f"  AUC={auc:.4f}  PCC={pcc:.4f}")
print(f"  t=0.5:  Rec={rec_05:.4f}  Prec={prec_05:.4f}  F1={f1_05:.4f}")
print(f"  max-F1: Rec={best_rec:.4f}  Prec={best_prec:.4f}  F1={best_f1:.4f} (t={best_t:.3f})")

result = {
    "detector": args.detector, "method": "combined_all",
    "auc": auc, "pcc": pcc, "n_total": n_total, "n_pos": n_pos,
    "rec_05": rec_05, "prec_05": prec_05, "f1_05": f1_05,
    "best_t": best_t, "rec_bf": best_rec, "prec_bf": best_prec, "f1_bf": best_f1,
}
save_path = f"./log/cross_chunk_{args.detector}_combined_scores.json"
with open(save_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {save_path}")
