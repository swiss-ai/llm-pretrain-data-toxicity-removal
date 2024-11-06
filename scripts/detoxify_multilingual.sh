#!/bin/bash
pip install -r requirements.txt

python scripts/detoxify_multilingual.py --model_dir /users/fsimin/multilingual_pretrain/detoxify_models/checkpoint_multilingual --language french
python scripts/detoxify_multilingual.py --model_dir /users/fsimin/multilingual_pretrain/detoxify_models/checkpoint_multilingual --language italian
python scripts/detoxify_multilingual.py --model_dir /users/fsimin/multilingual_pretrain/detoxify_models/checkpoint_multilingual --language russian
python scripts/detoxify_multilingual.py --model_dir /users/fsimin/multilingual_pretrain/detoxify_models/checkpoint_multilingual --language portuguese
python scripts/detoxify_multilingual.py --model_dir /users/fsimin/multilingual_pretrain/detoxify_models/checkpoint_multilingual --language spanish
python scripts/detoxify_multilingual.py --model_dir /users/fsimin/multilingual_pretrain/detoxify_models/checkpoint_multilingual --language turkish