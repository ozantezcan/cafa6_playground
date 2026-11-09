# cafa6_playground

Practice repo for the Kaggle [CAFA 6 protein function prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)
competition: predict Gene Ontology terms from protein sequences. Working log, findings and plan: [NOTES.md](NOTES.md).

## Resume after a pod restart

```bash
source /workspace/setup.sh   # restores bashrc, git identity, Claude Code, Claude memory, pip deps; checks CUDA
claude                       # then: "read NOTES.md and let's continue"
```

In VSCode (Remote-SSH): open the folder `/workspace/cafa6_playground` (not `/workspace`) and pick the
`/usr/local/bin/python` kernel for notebooks. Extensions are wiped on restart unless your *local* `settings.json` has
`"remote.SSH.defaultExtensions": ["ms-python.python", "ms-toolsai.jupyter", "mhutchie.git-graph"]`.

## Before stopping the pod

```bash
git status                   # commit + push anything you want to keep
```
Only `/workspace` survives a restart; everything in `/root` (home) is wiped.

## What lives where

| Path | What | In git? |
|---|---|---|
| `/workspace/cafa6_playground` | this repo (code, notebooks, NOTES.md, project `setup.sh`) | yes |
| `/workspace/setup.sh` | pod bootstrap, calls the repo's `setup.sh` | no (user-specific) |
| `/workspace/.secrets/` | `anthropic.env`, `git-credentials` | **never** |
| `/workspace/.kaggle/access_token` | Kaggle API token | **never** |
| `/workspace/data/cafa6` | competition data | no |
| `/workspace/embeddings` | protein embeddings | no |
| `/workspace/.claude-memory` | Claude Code memory (symlinked into `~/.claude`) | no |
| `/workspace/.cache` | Hugging Face models + pip cache | no |

## On a brand-new pod (empty `/workspace`)

1. `git clone https://github.com/ozantezcan/cafa6_playground /workspace/cafa6_playground`
2. Recreate `/workspace/.secrets/anthropic.env`, `/workspace/.secrets/git-credentials` and
   `/workspace/.kaggle/access_token` (paste them via the VSCode editor, never into chat or shell).
3. Recreate `/workspace/setup.sh` (ask Claude: "write the pod bootstrap described in the README"; it covers the
   bashrc lines, git identity, Claude Code install, the memory symlink, then `source`s the repo's `setup.sh`).
4. `source /workspace/setup.sh`, then download the data:
   `kaggle competitions download cafa-6-protein-function-prediction -p /workspace/data/cafa6` and unzip it.
