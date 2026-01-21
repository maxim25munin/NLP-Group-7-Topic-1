# Run `scripts/evaluate_ood_xlmr_vs_fasttext.py` on GPU

This guide walks through the full GPU setup for evaluating the XLM-R vs. fastText script, including environment prep, model downloads, and the GPU-specific run command.

## 1) Create and activate a Python environment

Pick your preferred virtual environment manager. For example, using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install GPU-enabled dependencies

Install PyTorch with CUDA support that matches your GPU driver/CUDA runtime. The PyTorch selector page (https://pytorch.org/get-started/locally/) provides the exact command. For example, CUDA 12.1 wheels:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install the script’s optional transformer stack and fastText dependencies:

```bash
pip install -r docs/requirements-transformers.txt
pip install fasttext scikit-learn pandas numpy
```

> **Note:** `fasttext` is required; the script exits if it is missing.

## 3) Download fastText language models

The script expects the binary fastText vectors in `models/fasttext/` with names like `cc.<code>.300.bin`. Create the directory and download the models for the five default languages (Kazakh, Latvian, Swedish, Yoruba, Urdu):

```bash
mkdir -p models/fasttext
cd models/fasttext

# Download fastText models (cc.<code>.300.bin)
for code in kk lv sv yo ur; do
  wget -c https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.${code}.300.bin.gz
  gunzip -f cc.${code}.300.bin.gz
 done
```

Return to the repository root:

```bash
cd ../..
```

## 4) Verify the data layout

The script’s defaults assume:

- Wikipedia CoNLL-U data under `data/<language>/*.conllu` (already present in this repo for the default languages).
- OOD CSVs already available (defaults are in the script; they map to `data/*_fasttext.csv`).

If you placed data elsewhere, pass explicit overrides (see step 6).

## 5) Confirm GPU visibility in PyTorch

Make sure PyTorch sees your GPU:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY
```

If `CUDA available: False`, double-check your driver installation and that your PyTorch wheel matches the local CUDA runtime.

## 6) Run the script on GPU

The script automatically uses the GPU through the Hugging Face `Trainer` when CUDA is available. Use the default configuration or pass overrides as needed.

**Default run (GPU used automatically):**

```bash
python scripts/evaluate_ood_xlmr_vs_fasttext.py
```

**Explicit GPU selection (e.g., use GPU 0 only):**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_ood_xlmr_vs_fasttext.py
```

**Example with custom paths and training overrides:**

```bash
python scripts/evaluate_ood_xlmr_vs_fasttext.py \
  --data-dir data \
  --fasttext-model-dir models/fasttext \
  --xlmr-model xlm-roberta-base \
  --xlmr-epochs 1 \
  --xlmr-batch-size 4 \
  --xlmr-learning-rate 5e-5 \
  --xlmr-weight-decay 0.01
```

## 7) (Optional) FastText-only run

If you only want the fastText baseline, skip XLM-R:

```bash
python scripts/evaluate_ood_xlmr_vs_fasttext.py --skip-xlmr
```

## 8) Output artifacts

- XLM-R checkpoints/logs: `reports/xlmr_ood_language_id/` (configurable via `--xlmr-output-dir`)
- Console output includes per-language OOD metrics and a summary table for fastText vs. XLM-R.

## Troubleshooting

- **Import error for `transformers` or `huggingface_hub`:** reinstall with `pip install -r docs/requirements-transformers.txt`.
- **OOM on GPU:** reduce `--xlmr-batch-size` (e.g., `--xlmr-batch-size 2`) or lower `--max-sentences`.
- **Missing fastText models:** verify the `models/fasttext/cc.<code>.300.bin` files exist and match the language codes passed via `--fasttext-language-codes`.
