# Experimental Environment

This project used two software environments: a local macOS environment for
data processing, EDA, and the LLaMA/Ollama experiments, and a Google Colab
Linux environment for the RoBERTa baseline.

A single `requirements.txt` is provided. It uses Python environment markers
(`sys_platform`) so that the appropriate package versions are installed on
macOS or Linux.

## Local LLaMA / EDA environment

- Operating environment: macOS
- Python: 3.13.9
- Ollama application/server: 0.31.1
- Ollama Python client: 0.6.2
- NumPy: 1.26.4
- pandas: 2.3.3
- Matplotlib: 3.10.7
- scikit-learn: 1.8.0
- tqdm: 4.67.3
- requests: 2.32.5
- Inference: local Ollama
- LLaMA temperature: 0

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The exact prompt templates used for the three LLaMA conditions are stored in
the `prompts/` directory.

## RoBERTa / Google Colab environment

- Operating environment: Google Colab (Linux)
- Python: 3.13.15
- PyTorch: 2.11.0+cu128
- Transformers: 5.15.1
- Datasets: 4.0.0
- Accelerate: 1.14.0
- NumPy: 2.1.3
- pandas: 2.2.3
- scikit-learn: 1.6.1
- Matplotlib: 3.10.0
- Runtime shown in Colab: NVIDIA T4 GPU

The PyTorch requirement is written as `torch==2.11.0` in `requirements.txt`.
The exact Colab runtime reported the CUDA-enabled build `2.11.0+cu128`, which
is recorded here because CUDA build selection can depend on the installation
source and runtime image.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## RoBERTa experimental configuration

The final RoBERTa baseline used:

- Base checkpoint: `roberta-base`
- Number of output labels: 2
- Maximum sequence length: 128
- Dynamic padding
- Training batch size: 16
- Evaluation batch size: 16
- Learning rate: 2e-5
- Epochs: 3
- Weight decay: 0.01
- Random seed: 42
- Evaluation and checkpoint saving: once per epoch
- Best checkpoint criterion: validation positive-class F1
- Load best model at end: yes
- Explicit class weighting: none
- Final inference batch size: 32

## Reproducibility note

The macOS versions were captured from the local environment used for the
LLaMA/EDA workflow.

The Colab package versions were captured from the RoBERTa notebook runtime.
If the runtime shown when these versions were recorded was not the original
training runtime, these versions should be treated as the documented
reproduction environment rather than claimed as exact historical package
versions of the original training run.

GPU execution can still introduce some nondeterminism even with fixed random
seeds. The repository therefore also preserves the data-processing scripts,
fixed split construction, prompts, evaluation scripts, aggregate results, and
RoBERTa training configuration.
