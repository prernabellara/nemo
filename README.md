# NVIDIA NeMo Sample Project

This repository demonstrates how to build and fine-tune a GPT-style language model using the NVIDIA NeMo framework.

## Features
- GPT model fine-tuning with NeMo
- Hydra config system
- Preprocessing placeholders
- Scripted data download

## Quick Start
```bash
pip install -r requirements.txt
bash scripts/download_data.sh
python nemo_examples/train_gpt.py model.train_ds.file_path=data/sample.txt
```
