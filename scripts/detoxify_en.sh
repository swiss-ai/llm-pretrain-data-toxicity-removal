#!/bin/bash
pip install -r requirements.txt

python scripts/detoxify_multilingual.py --model_dir /users/fsimin/multilingual_pretrain/detoxify_models/checkpoint_en 
