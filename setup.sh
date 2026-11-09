#!/usr/bin/env bash
# Project environment for cafa6_playground (pod-level setup lives in /workspace/setup.sh, which calls this).
# Usage: source /workspace/cafa6_playground/setup.sh
# Kaggle CLI 2.x: KAGGLE_API_TOKEN may be a *path* to the token file (keeps the secret out of env/history).
export KAGGLE_API_TOKEN=/workspace/.kaggle/access_token
# torch comes from the pod image; requirements.txt deliberately doesn't list it.
pip install -q -r /workspace/cafa6_playground/requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda OK')"
[ -d /workspace/data/cafa6 ] || echo "NOTE: data missing -> kaggle competitions download cafa-6-protein-function-prediction -p /workspace/data/cafa6"
