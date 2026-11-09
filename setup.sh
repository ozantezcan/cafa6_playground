#!/usr/bin/env bash
# Restore environment after pod restart (home dir + pip installs are not persistent).
# Usage: source /workspace/cafa6_playground/setup.sh
# Kaggle CLI 2.x: KAGGLE_API_TOKEN may be a *path* to the token file (keeps the secret out of env/history).
export KAGGLE_API_TOKEN=/workspace/.kaggle/access_token
pip install -q -r /workspace/cafa6_playground/requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda OK')"
