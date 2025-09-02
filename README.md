
# 🔥 Open-dCoder: Open Diffusion Large Language Model


👉 *Train it. Evaluate it. Run it. Reproduce it.*

The **most open release of a diffusion large language model** for code generation to date. —  including pretraining, evaluation, inference, and checkpoints.
## 🎥 Demo

<p align="center">
  <img src="https://github.com/pengzhangzhi/dLLM-training/blob/main/assets/quick-sort-demo.gif" 
       alt="Quick Sort Demo" width="600"/>
</p>

<p align="center">
  <a href="https://youtu.be/d8WrmvUhO9g">
    <img src="https://img.shields.io/badge/YouTube-Video-red?logo=youtube" alt="YouTube link"/>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.bilibili.com/video/BV1ZveSz3E1J/">
    <img src="https://img.shields.io/badge/Bilibili-视频-blue?logo=bilibili" alt="Bilibili link"/>
  </a>
</p>


## Why Open-dCoder?

Most diffusion LLM repos (e.g., LLaDA, Dream) only release **inference scripts + weights**, which limits reproducibility.  
**Open-dCoder** is the first to open-source the **entire stack**:

- 🏋️ **Pretraining code + data** — train your own diffusion LLMs from scratch  
- ⚡ **Inference scripts** — run generations and benchmarks easily  
- 📊 **Evaluation suite** — lm-eval-harness + custom metrics for full reproducibility  
- 📦 **Weights + checkpoints** — Hugging Face uploads for direct use  

👉 With Open-dCoder, you can go from raw data → training → checkpoints → evaluation → inference, all in one repo.

###  Transparency Comparison of Diffusion LLM Releases

| Project                                                                 | Data | Training Code | Inference | Evaluation | Weights |
|-------------------------------------------------------------------------|:---:|:-------------:|:---------:|:----------:|:-------:|
| **Open-dCoder (ours)**                                                  | ✅  | ✅            | ✅        | ✅         | ✅      |
| [LLaDA](https://github.com/ML-GSAI/LLaDA)                               | ❌  | ❌            | ✅        | ⚠️ Limited | ✅      |
| [Dream](https://github.com/HKUNLP/Dream)                                | ❌  | ❌            | ✅        | ✅         | ✅      |
| [Gemini-Diffusion](https://deepmind.google/models/gemini-diffusion/)    | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |
| [Seed Diffusion](https://seed.bytedance.com/seed_diffusion)             | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |
| [Mercury](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model) | ❌  | ❌            | ❌        | ❌         | ❌ (API only) |


### Install
We use `micromamba` for Env management, feel free to revise it to `conda`:
```bash

micromamba install -c nvidia/label/cuda-12.3.0 cuda-toolkit -y
pip install ninja

# install the newest torch with cu121
pip install torch==2.5.0  --index-url https://download.pytorch.org/whl/cu121

pip install "flash-attn==2.7.4.post1" \
--extra-index-url https://github.com/Dao-AILab/flash-attention/releases/download


pip install --upgrade --no-cache-dir \
  tensordict torchdata byte-flux triton>=3.1.0 \
  transformers==4.54.1 accelerate datasets peft hf-transfer \
  codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc \
  wandb ninja liger-kernel \
  pytest yapf py-spy pyext pre-commit ruff packaging

pip install -e .

```

## 🚀 Quickstart: Sampling

Once installed (see below), you can try code generation in a few lines:

```python
from transformers import AutoTokenizer
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig
import torch

model_id = "fredzzp/open-dcoder-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = Qwen2ForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()

# Prompt
prompt = "Write a quick sort algorithm in python."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generation config
gen_cfg = MDMGenerationConfig(max_new_tokens=128, steps=200, temperature=0.7)

with torch.no_grad():
    outputs = model.diffusion_generate(inputs=input_ids, generation_config=gen_cfg)

print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
```

👉 For full logging, history tracking, and file output, run:
```bash
python sample.py
```

### 📊 Benchmarking

We release a fully open-source **evaluation suite** for diffusion-based LLMs (dLLMs), covering both **standard code generation tasks** and **code infilling tasks**.

Benchmarks include:

* **HumanEval / HumanEval+**
* **MBPP / MBPP+**
* **HumanEval-Infill**
* **SantaCoder-FIM**

#### standard code generation tasks
  
| Method                   | HumanEval        |                   | HumanEval+       |                   | MBPP           |                   | MBPP+          |                   |
|---------------------------|------------------|-------------------|------------------|-------------------|----------------|-------------------|----------------|-------------------|
|                           | Pass@1           | Pass@10           | Pass@1           | Pass@10           | Pass@1         | Pass@10           | Pass@1         | Pass@10           |
| LLaDA (8B)                | 35.4             | 50.0                 | 30.5             | 43.3                 | 50.1           | –                 | 42.1           | –                 |
| Dream (7B)                | 56.7             | 59.2                 | 50.0             | 53.7                 | 68.7           | –                 | 57.4           | –                 |
| Mask DFM (1.3B)           | 9.1              | 17.6              | 7.9              | 13.4               | 6.2            | 25.0              | –              | –                 |
| Edit Flow (1.3B)          | 12.8             | 24.3              | 10.4             | 20.7              | 10.0           | 36.4              | –              | –                 |
| **Open-dCoder (0.5B, Ours)**  | 20.8             | 38.4              | 17.6             | 35.2              | 16.7           | 38.4              | 23.9           | 53.6              |

#### code infilling tasks

| Method | HumanEval Infill Pass@1 | SantaCoder Exact Match |
|--------------------------------|--------------------------|-------------------------|
| LLaDA-8B                       | 48.3                     | 35.1                    |
| Dream-7B                       | 39.4                     | 40.7                    |
| DiffuCoder-7B                  | 54.8                     | 38.8                    |
| Dream-Coder-7B                 | 55.3                     | 40.0                    |
| **Open-dCoder (0.5B, Ours)**       | 77.4                     | 56.4                    |
 
### Evaluation

Installing the following local pkgs 
```
pip install -e lm-evaluation-harness human-eval-infilling
```
#### Code Completion Evaluation (Humaneval and MBPP)
```
cd eval/eval_infill
bash run_eval.sh
```
#### Code Infilling Evaluation 
```
cd eval/eval_infill
bash run_eval.sh
```


## Pretraining

* **Data**: We prepare a concise, high-quality code corpus, **[FineCode](https://huggingface.co/)**, hosted openly on Hugging Face.
* **Initialization**: Following the approach in *Dream*, we continue pretraining from an existing autoregressive model, **Qwen2.5-Coder**, adapting it into the diffusion framework.

### Download data

```bash
python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --local_dir ./data
```



### Training

```bash
python3 tasks/train_torch.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --data.train_path=data/data \
  --train.ckpt_manager=dcp \
  --train.micro_batch_size=16 \
  --train.global_batch_size=512 \
  --train.output_dir=logs/Qwen2.5-Coder-0.5B_mdm \
  --train.save_steps=10000
```

### Uploading Checkpoints to Huggingface

```bash
from huggingface_hub import HfApi

REPO_ID = "fredzzp/open-dcoder-0.5B"
LOCAL_DIR = "logs/Qwen2.5-Coder-0.5B_mdm/checkpoints/global_step_370000/hf_ckpt"

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=REPO_ID, repo_type="model", folder_path=LOCAL_DIR)

```


## 🙏 Appreciation

No single line of this project would be possible without the incredible work from the open research community. We gratefully acknowledge the following contributions and inspirations:

### ⚙️ Frameworks & Tooling

* [**VeOmni**](https://github.com/ByteDance-Seed/VeOmni) — pretraining framework we build upon.
* [**lm-evaluation-harness**](https://github.com/EleutherAI/lm-evaluation-harness) — standard evaluation suite we adopt and extend.

### 🌊 Open-Source dLLM Projects

* [**LLaDA**](https://github.com/ML-GSAI/LLaDA) 
* [**Dream**](https://github.com/HKUNLP/Dream)

### 🚀 Pioneering Diffusion LLM Works

* [**Gemini-Diffusion**](https://deepmind.google/models/gemini-diffusion/) (DeepMind)
* [**Seed Diffusion**](https://seed.bytedance.com/seed_diffusion) (ByteDance)
* [**Mercury**](https://www.inceptionlabs.ai/introducing-mercury-our-general-chat-model) (InceptionLabs)

### 📖 Foundational Research on Masked Diffusion Models

* Jiaxin Shi — [**MD4**](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html)
* Sahoo et al. — [**MDLM**](https://arxiv.org/abs/2406.07524)
* Zaixiang et al. — [**DPLM**](https://github.com/bytedance/dplm)



