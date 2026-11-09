# CAFA 6 – working notes

## 2026-11-08 – Environment
- RunPod RTX 4090 24GB, Python 3.12.3, torch 2.8.0+cu128 (preinstalled; never pip-install torch).
- Installed: pandas 3.1.1, scikit-learn 1.9.1, transformers 5.22.2, kaggle 2.2.5, matplotlib, jupyter, tqdm. numpy stayed 2.5.2 (checked with `pip install --dry-run` first).
- Removed torch/torchvision from requirements.txt so `pip install -r` can't replace the pod build.
- Persistence: only /workspace survives restarts -> `source /workspace/cafa6_playground/setup.sh` after restart.
- Kaggle creds: CLI 2.2.5 uses a single API token (not kaggle.json) and only auto-reads ~/.kaggle/access_token
  (wiped on restart). So token lives at /workspace/.kaggle/access_token (chmod 600) and setup.sh sets
  KAGGLE_API_TOKEN to that *path* (the CLI accepts a file path there).
- Data -> /workspace/data/cafa6, embeddings -> /workspace/embeddings.

## 2026-11-08 – Data + EDA (notebooks/01_eda.ipynb)
- Downloaded via `kaggle competitions download cafa-6-protein-function-prediction` -> /workspace/data/cafa6 (~200MB unzipped).
- Train: 82,404 proteins, 537k annotations, 26k terms (BP 16.9k / MF 6.6k / CC 2.7k). Every train protein has >=1 label.
- **Labels are NOT propagated** (0% of annotations have full ancestor set, no roots). Propagating (is_a+part_of)
  grows labels 6.6x. -> must propagate for training targets and for eval ground truth.
- **All 82k train proteins are also in the 224k test superset** (36.7% of test). Test also scores new annotations on
  known proteins -> known train labels should go into the submission for those IDs. Also: only need to embed the
  test superset (it contains train).
- Lengths: median 409 (train) / 342 (test); 9.3% train / 6.0% test > 1022 aa -> need truncation or chunking for ESM-2.
- Top-K coverage of raw annotations: CC saturates (top 500 = 91%), MF top 1000 = 83%, BP top 2000 = only 63%.
  (Computed on unpropagated labels; recompute after propagation.)
- IA weights: 29.5% of terms have IA=0. In BP, only ~12% of IA-weighted mass sits in terms with >200 proteins ->
  frequency baseline will be weak on BP; CC is dominated by frequent terms.
- Species: train 1,381, test 8,453 (7,072 unseen in train). Model organisms dominate both.
- sample_submission contains `Text` rows (protein ID, "Text", score, free-text description) — new in CAFA 6.
  TODO: read the competition Evaluation page for whether/how Text rows are scored. Ignore for now.
- Git: data stays out of the repo (outside repo dir + gitignore patterns for *.fasta, *.obo, *.zip, *.npy, submissions).
  Kaggle data shouldn't be redistributed anyway.

## Resume checklist (after pod restart / new session)
1. `source /workspace/cafa6_playground/setup.sh` (reinstalls pip deps, sets KAGGLE_API_TOKEN, checks CUDA).
2. If /workspace/data/cafa6 is missing: `kaggle competitions download cafa-6-protein-function-prediction -p /workspace/data/cafa6` + unzip.
   If /workspace/.kaggle/access_token is missing: new token at kaggle.com/settings/api, save it there via the VSCode editor
   (never paste into chat/shell).
3. VSCode: the Jupyter + Python extensions live in ~/.vscode-server (wiped on restart). Add
   `"remote.SSH.defaultExtensions": ["ms-python.python", "ms-toolsai.jupyter"]` to *local* settings.json.
   Kernel = /usr/local/bin/python (no separate env: a venv wouldn't have the pod's torch). Use `%pip`, not `!pip`.
4. Notebooks are paired with jupytext (`notebooks/*.py` <-> `.ipynb`); edit either, `jupytext --sync` to update.

## Concepts learned (2026-11-09)
- Protein = string over 20 amino-acid letters; length = number of residues. Rare extras: U, O, X, B, Z.
- GO = DAG of ~40k terms in 3 aspects: MF (what it does), BP (which process), CC (where). True path rule:
  a term implies all its ancestors. Roots (GO:0003674 MF, GO:0008150 BP, GO:0005575 CC) carry no info, not scored.
- CAFA is prospective: scored only on annotations that appear *after* the deadline. Metric = IA-weighted F-max per
  aspect, averaged. IA ≈ -log2 P(term | parents): specific terms are worth more. F-max uses the best global threshold,
  so scores need good ranking, not calibration.
- ESM-2 window: trained on <=1024 tokens = <cls> + 1022 residues + <eos>. It uses rotary position embeddings, so the
  limit is soft (longer inputs run, but they weren't seen in training and attention memory grows with the square of
  length). Options: truncate (v1) or chunk + average.
- Model organisms (human, mouse, Arabidopsis, yeast, ...) = 82% of train but only ~40% of test; 7k test species are
  unseen in training -> rely on sequence features. Scored targets are probably skewed back toward model organisms (unverified).

## Method landscape (brainstorm 2026-11-09)
- Baselines: frequency ("Naive"), homology transfer via BLAST/DIAMOND (official CAFA baselines; homology is strong).
- PLM embeddings (ESM-2, ProtT5, Ankh, ESM-C) + MLP head per aspect; embedding kNN (goPredSim-style).
- Heavier: LoRA fine-tuning of the PLM; structure (AlphaFold/Foldseek/DeepFRI), InterPro domains, PPI networks.
- Label tricks: propagate predictions up the DAG (parent = max of children), per-aspect top-K, focal/asymmetric or
  IA-weighted loss, reuse known labels for train proteins that are also in test.
- Ensembling: per-aspect blend of frequency + homology + PLM head + kNN.
- Validation options: random split (optimistic, homologs leak), cluster-based split (MMseqs2 clusters at 30-50% identity,
  realistic), time-based split (old vs new GOA release, most faithful, costly). Select on per-aspect val F-max, not loss.
  Cross-check our metric against the official CAFA-evaluator (`cafaeval`). TODO: check CAFA 6 rules for per-protein
  prediction caps and how the `Text` rows are scored.

## Agreed plan (2026-11-09)
| # | Step | Status |
|---|---|---|
| 0-1 | Housekeeping, deps | ✅ |
| 2-3 | Data download, EDA notebook | ✅ |
| 4 | IA-weighted F-max (+ check vs cafaeval) + frequency baseline, random 10% split | next |
| 5 | Known-label transfer for test proteins already in train | |
| 6 | DIAMOND homology baseline (binary download, not pip) | |
| 7 | ESM-2 650M embeddings (embed the test superset only; it contains train), smoke-test on ~1k seqs first | |
| 8 | MLP head per aspect + embedding kNN | |
| 9 | Per-aspect blend + GO propagation | |
| 10 | Cluster-based split, then extras (ProtT5, chunking, asymmetric loss) | |

## Open decisions
- Random 10% split vs building the cluster-based split from the start (leaning: random first, cluster at step 10).
- DIAMOND in scope? (proposed yes)
