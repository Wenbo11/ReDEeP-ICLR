# Codebase Modifications for Reproduction

Date: 2026-04-18

This document catalogues every change made to the ReDeEP-ICLR repository to
get the RAGTruth experiments running on a fresh machine (Ubuntu, NVIDIA RTX PRO
6000 Blackwell, no pre-existing Python environment).

---

## 1. Environment setup

The original repo provides a `requirements.txt` in conda format. We replaced
it with a `uv`-managed virtualenv:

```bash
uv venv --python 3.11 .venv
uv pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install accelerate datasets sentence-transformers scikit-learn scipy pandas numpy tqdm huggingface-hub tokenizers
uv pip install -e transformers/   # repo's modified HuggingFace transformers
```

PyTorch nightly (cu128) was required because the Blackwell GPU (sm_120) is not
supported by stable PyTorch releases as of this date.

---

## 2. Changes to `ReDeEP/token_level_detect.py`

### 2a. Model paths (hardcoded -> HuggingFace IDs)

The original code loads models from a lab-specific absolute path
`/home/sunhao_dai/PLMs/...`. Changed to standard HuggingFace model IDs so
`from_pretrained` resolves them from the HuggingFace cache automatically.

```diff
-    model_name = "llama2/llama-2-7b-chat-hf"
+    model_name = "meta-llama/Llama-2-7b-chat-hf"

-    model_name = "llama2/llama-2-13b-chat-hf"
+    model_name = "meta-llama/Llama-2-13b-chat-hf"

-    model_name = "llama3/Meta-Llama-3-8B-Instruct/"
+    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

-model = AutoModelForCausalLM.from_pretrained(
-    f"/home/sunhao_dai/PLMs/{model_name}", ...)
-tokenizer = AutoTokenizer.from_pretrained(f"/home/sunhao_dai/PLMs/{model_name}")
+model = AutoModelForCausalLM.from_pretrained(model_name, ...)
+tokenizer = AutoTokenizer.from_pretrained(model_name)

-    tokenizer_for_temp = AutoTokenizer.from_pretrained("/home/sunhao_dai/PLMs/llama2/llama-2-7b-chat-hf")
+    tokenizer_for_temp = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
```

### 2b. Dataset paths (missing subdirectory)

The dataset files live under `dataset/dataset/` but the code expected them
directly under `dataset/`.

```diff
-        response_path = "../dataset/response_with_llama3_8b.jsonl"
+        response_path = "../dataset/dataset/response_with_llama3_8b.jsonl"

-        response_path = "../dataset/response.jsonl"
+        response_path = "../dataset/dataset/response.jsonl"

-    response_path = "../dataset/response_dolly.jsonl"
+    response_path = "../dataset/dataset/response_dolly.jsonl"

-    source_info_path = "../dataset/source_info.jsonl"
+    source_info_path = "../dataset/dataset/source_info.jsonl"

-    source_info_path = "../dataset/source_info_dolly.jsonl"
+    source_info_path = "../dataset/dataset/source_info_dolly.jsonl"
```

Also collapsed a redundant conditional (both branches of the `llama3-8b` /
else check pointed to the same `source_info.jsonl`).

### 2c. Attention implementation

The repo's modified `modeling_llama.py` adds custom keyword arguments
(`disturb_head_ids`, `select_heads_ids`, etc.) to `LlamaAttention.forward()`.
When HuggingFace auto-selects `attn_implementation="sdpa"` (the default),
`LlamaSdpaAttention` is used instead. Its `forward()` does fall back to
`super().forward()` when `output_attentions=True`, but it only forwards the
standard kwargs, dropping the custom ones and crashing with:

```
TypeError: LlamaSdpaAttention.forward() got an unexpected keyword argument 'disturb_head_ids'
```

Fix: force eager attention at model load time.

```diff
 model = AutoModelForCausalLM.from_pretrained(
     model_name,
     device_map="auto",
-    torch_dtype=torch.float16
+    torch_dtype=torch.float16,
+    attn_implementation="eager"
 )
```

### 2d. Input tensors not on GPU

With `device_map="auto"`, the model lives on CUDA, but `tokenizer(...)` returns
CPU tensors. The original code worked on the authors' setup (possibly
`torch<2.0` or different device placement), but current PyTorch raises:

```
RuntimeError: Expected all tensors to be on the same device, but got index is on cpu
```

Fix: move `input_ids` and `prefix_ids` to the model's device.

```diff
-        input_ids = tokenizer([input_text], return_tensors="pt").input_ids
-        prefix_ids = tokenizer([text], return_tensors="pt").input_ids
+        input_ids = tokenizer([input_text], return_tensors="pt").input_ids.to(device)
+        prefix_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
```

---

## 3. Changes to `ReDeEP/chunk_level_detect.py`

All four categories of fixes from section 2 apply identically (model paths,
dataset paths, attention implementation, input tensor device). In addition:

### 3a. BGE embedding model path

```diff
-bge_model = SentenceTransformer('/home/zhongxiang_sun/code/LLMs/bge-base-en-v1.5/').to("cuda:0")
+bge_model = SentenceTransformer('BAAI/bge-base-en-v1.5').to("cuda:0")
```

Loads from HuggingFace cache instead of a lab-specific path.

---

## 4. Changes to `ReDeEP/token_level_reg.py`

### 4a. DataFrame slicing bug

```diff
-    auc_external_similarity, auc_parameter_knowledge_difference = calculate_auc_pcc(df.iloc[:, :int(df.shape[1] * 0.5)], number)
+    auc_external_similarity, auc_parameter_knowledge_difference = calculate_auc_pcc(df, number)
```

**Root cause:** The dataframe has 66 columns (`identifier` + 32
`external_similarity_*` + 32 `parameter_knowledge_difference_*` +
`hallucination_label`). Slicing to the first 33 columns drops
`parameter_knowledge_difference_*` and `hallucination_label`, which
`calculate_auc_pcc` needs. The function already selects columns by name, so
passing the full dataframe is correct.

### 4b. Added Recall/F1 computation

The paper's Table 1 reports Recall and F1, but the original code only computed
AUC and PCC. Added a threshold sweep to `calculate_auc_pcc_32_32`:

```python
best_f1 = 0
best_threshold = 0.5
best_recall = 0
for t in np.arange(0.01, 1.0, 0.01):
    preds = (grouped_df['difference_normalized_mean_norm'] >= t).astype(int)
    f1_val = f1_score(grouped_df['hallucination_label'], preds, zero_division=0)
    if f1_val > best_f1:
        best_f1 = f1_val
        best_threshold = t
        best_recall = recall_score(grouped_df['hallucination_label'], preds, zero_division=0)

return auc_difference_normalized, person_difference_normalized, best_recall, best_f1
```

Updated the caller and output dict to include the new metrics:

```diff
-    result_dict = {"auc":auc_difference_normalized, "pcc": person_difference_normalized}
+    result_dict = {"auc": auc_difference_normalized, "pcc": person_difference_normalized, "recall": best_recall, "f1": best_f1}
```

---

## 5. Changes to `ReDeEP/chunk_level_reg.py`

### 5a. Same DataFrame slicing fix as 4a

```diff
-    auc_external_similarity, _, auc_parameter_knowledge_difference, _ = calculate_auc_pcc(df.iloc[:, :int(df.shape[1] * 0.5)], ext_map_dict, para_map_dict, number)
+    auc_external_similarity, _, auc_parameter_knowledge_difference, _ = calculate_auc_pcc(df, ext_map_dict, para_map_dict, number)
```

### 5b. Same Recall/F1 addition as 4b

Identical threshold sweep logic added to `calculate_auc_pcc_32_32`, with
matching return-value and output-dict changes.

### 5c. Dolly dataset path bug (copy-paste)

Two `elif` branches checked for `"ragtruth"` instead of `"dolly"`, making the
Dolly data path unreachable:

```diff
     if args.model_name == "llama2-7b":
         if args.dataset == "ragtruth":
             data_path = "./log/test_llama2_7B/llama2_7B_response_chunk.json"
-        elif args.dataset == "ragtruth":
+        elif args.dataset == "dolly":
             data_path = "./log/test_llama2_7B/llama2_7B_response_chunk_dolly.json"

     elif args.model_name == "llama2-13b":
         if args.dataset == "ragtruth":
             data_path = "./log/test_llama2_13B/llama2_13B_response_chunk.json"
-        elif args.dataset == "ragtruth":
+        elif args.dataset == "dolly":
             data_path = "./log/test_llama2_13B/llama2_13B_response_chunk_dolly.json"
```

### 5d. Dataset path (same as section 2b)

```diff
-source_info_path = "../dataset/source_info.jsonl"
+source_info_path = "../dataset/dataset/source_info.jsonl"
```

---

## Summary of changes by category

| Category | Files affected | Lines changed |
|----------|---------------|---------------|
| Hardcoded model paths to HF IDs | `token_level_detect.py`, `chunk_level_detect.py` | ~12 each |
| Dataset directory paths | all 4 scripts | ~5 each |
| `attn_implementation="eager"` | `token_level_detect.py`, `chunk_level_detect.py` | 1 each |
| `.to(device)` on input tensors | `token_level_detect.py`, `chunk_level_detect.py` | 2 each |
| BGE model path | `chunk_level_detect.py` | 1 |
| DataFrame `.iloc` slicing bug | `token_level_reg.py`, `chunk_level_reg.py` | 1 each |
| Dolly `elif` copy-paste bug | `chunk_level_reg.py` | 2 |
| Added Recall/F1 metrics | `token_level_reg.py`, `chunk_level_reg.py` | ~15 each |

No changes were made to the modified `transformers/` library, the dataset
files, or the pre-computed `topk_heads.json` / `token_hyperparameter.json`
files.
