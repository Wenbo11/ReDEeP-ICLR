"""Score cross-model chunk-level detection results and produce pooled metrics."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--detector', type=str, required=True)
parser.add_argument('--generators', type=str, nargs='+', required=True)
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


def construct_dataframe(file_path, number):
    with open(file_path, "r") as f:
        response = json.load(f)

    data_dict = {
        "identifier": [],
        "type": [],
        **{f"external_similarity_{k}": [] for k in range(number)},
        **{f"parameter_knowledge_difference_{k}": [] for k in range(number)},
        "hallucination_label": []
    }

    ext_map_dict, para_map_dict = {}, {}
    for i, resp in enumerate(response):
        if resp["split"] != "test":
            continue
        respond_ids = resp["source_id"]
        rep_type = source_info_dict[respond_ids]["task_type"]

        for j in range(len(resp["scores"])):
            data_dict["identifier"].append(f"response_{i}_item_{j}")
            data_dict["type"].append(rep_type)
            for k in range(number):
                data_dict[f"external_similarity_{k}"].append(
                    list(resp["scores"][j]["prompt_attention_score"].values())[k]
                )
                data_dict[f"parameter_knowledge_difference_{k}"].append(
                    list(resp["scores"][j]["parameter_knowledge_scores"].values())[k]
                )
            data_dict["hallucination_label"].append(resp["scores"][j]["hallucination_label"])
        if i == len(response) - 1:
            ext_map_dict = {
                f"external_similarity_{k}": list(resp["scores"][j]["prompt_attention_score"].keys())[k]
                for k in range(number)
            }
            para_map_dict = {
                f"parameter_knowledge_difference_{k}": list(resp["scores"][j]["parameter_knowledge_scores"].keys())[k]
                for k in range(number)
            }

    return pd.DataFrame(data_dict), ext_map_dict, para_map_dict


def score_single(file_path, number, top_n, top_k, alpha, m=1):
    df, ext_map_dict, para_map_dict = construct_dataframe(file_path, number)

    auc_ext = []
    auc_param = []
    for k in range(number):
        auc_e = roc_auc_score(1 - df['hallucination_label'], df[f'external_similarity_{k}'])
        auc_ext.append((auc_e, f'external_similarity_{k}'))
        auc_p = roc_auc_score(df['hallucination_label'], df[f'parameter_knowledge_difference_{k}'])
        auc_param.append((auc_p, f'parameter_knowledge_difference_{k}'))

    top_ext = sorted(auc_ext, reverse=True)[:top_n]
    top_par = sorted(auc_param, reverse=True)[:top_k]

    df['ext_sum'] = df[[c for _, c in top_ext]].sum(axis=1)
    df['param_sum'] = df[[c for _, c in top_par]].sum(axis=1)

    scaler = MinMaxScaler()
    df['ext_norm'] = scaler.fit_transform(df[['ext_sum']])
    df['param_norm'] = scaler.fit_transform(df[['param_sum']])
    df['diff'] = m * df['param_norm'] - alpha * df['ext_norm']

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

    return {
        "auc": auc, "pcc": pcc,
        "n_total": n_total, "n_pos": n_pos,
        "rec_05": rec_05, "prec_05": prec_05, "f1_05": f1_05,
        "best_t": best_t, "rec_bf": best_rec, "prec_bf": best_prec, "f1_bf": best_f1,
    }, grouped


if __name__ == "__main__":
    all_results = []
    all_grouped = []

    for gen in args.generators:
        safe_gen = gen.replace("/", "_")
        path = f"./log/cross_chunk_{args.detector}_gen_{safe_gen}.json"
        print(f"\n=== {args.detector} detecting {gen} ===")
        try:
            result, grouped = score_single(path, args.number, args.top_n, args.top_k, args.alpha)
            result["generator"] = gen
            result["detector"] = args.detector
            all_results.append(result)
            all_grouped.append(grouped)
            print(f"  AUC={result['auc']:.4f}  PCC={result['pcc']:.4f}  F1(0.5)={result['f1_05']:.4f}  F1*={result['f1_bf']:.4f} (t={result['best_t']:.2f})  N={result['n_total']}  pos={result['n_pos']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if len(all_grouped) > 1:
        pooled = pd.concat(all_grouped, ignore_index=True)
        mn, mx = pooled['score'].min(), pooled['score'].max()
        pooled['score_renorm'] = (pooled['score'] - mn) / (mx - mn)

        auc = roc_auc_score(pooled['hallucination_label'], pooled['score_renorm'])
        pcc, _ = pearsonr(pooled['hallucination_label'], pooled['score_renorm'])
        n_total = len(pooled)
        n_pos = int(pooled['hallucination_label'].sum())

        preds_05 = (pooled['score_renorm'] >= 0.5).astype(int)
        rec_05 = recall_score(pooled['hallucination_label'], preds_05, zero_division=0)
        prec_05 = precision_score(pooled['hallucination_label'], preds_05, zero_division=0)
        f1_05 = f1_score(pooled['hallucination_label'], preds_05, zero_division=0)

        best_f1, best_t, best_rec, best_prec = 0, 0.5, 0, 0
        for t in np.arange(0.01, 1.0, 0.01):
            p = (pooled['score_renorm'] >= t).astype(int)
            f = f1_score(pooled['hallucination_label'], p, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = t
                best_rec = recall_score(pooled['hallucination_label'], p, zero_division=0)
                best_prec = precision_score(pooled['hallucination_label'], p, zero_division=0)

        print(f"\n=== POOLED ({args.detector}, {n_total} responses, {n_pos} hallucinated) ===")
        print(f"  AUC={auc:.4f}  PCC={pcc:.4f}")
        print(f"  t=0.5:  Rec={rec_05:.4f}  Prec={prec_05:.4f}  F1={f1_05:.4f}")
        print(f"  max-F1: Rec={best_rec:.4f}  Prec={best_prec:.4f}  F1={best_f1:.4f} (t={best_t:.3f})")

        pooled_result = {
            "detector": args.detector, "generator": "ALL_POOLED",
            "auc": auc, "pcc": pcc, "n_total": n_total, "n_pos": n_pos,
            "rec_05": rec_05, "prec_05": prec_05, "f1_05": f1_05,
            "best_t": best_t, "rec_bf": best_rec, "prec_bf": best_prec, "f1_bf": best_f1,
        }
        all_results.append(pooled_result)

    save_path = f"./log/cross_chunk_{args.detector}_all_scores.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all results to {save_path}")
