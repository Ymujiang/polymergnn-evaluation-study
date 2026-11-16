# PolymerGNN evaluation study — reproduction package

Code, figure scripts and result records for the protocol and split-sensitivity evaluation of
PolymerGNN on the polyester dataset.

## Licences

* Code in `scripts/` and `figures/`: MIT, see `LICENSE`.
* Result data in `evidence/` and `runtime_source_manifest.json`: CC BY 4.0, see `LICENSE-DATA`.

## Upstream

Upstream PolymerGNN is https://github.com/owencqueen/PolymerGNN at commit
`c0345c1476f137721039c3d7d8e6772d93075794`. That repository carries no licence grant, so its source is not
redistributed here. `polymergnn_compatibility.patch` and `runtime_source_manifest.json` let you rebuild the
pinned evaluation runtime from your own checkout of upstream.

## Layout

| Path | Contents |
|---|---|
| `scripts/analyze_r8.py` | Sealed held-out test comparison: per-seed losses, paired statistics, BCa intervals |
| `scripts/analyze_r13_val_test.py` | Paired validation–test association and leave-one-seed-out ranges |
| `scripts/analyze_multisplit.py` | Split-variance versus seed-variance comparison over 20 development partitions |
| `scripts/analyze_multiseed.py` | Repeated-training dispersion |
| `scripts/analyze_r9.py` | Ablation comparisons |
| `scripts/analyze_round2.py` | Second-round development statistics |
| `scripts/diagnose_r8_joint_baseline.py`, `scripts/diagnose_r8_joint_baseline_persample.py` | Joint-baseline failure diagnosis |
| `scripts/sealed_test_eval.py`, `scripts/independent_score.py`, `scripts/validate_evaluation.py` | Sealed test-set evaluation of saved checkpoints, independent rescoring of the exported predictions, provenance validation |
| `scripts/evaluate_smoke.py` | Single-run smoke test of the training and evaluation path; it calls a training entry point that is not part of this package |
| `scripts/materialize_runtime.py` | Rebuild the pinned upstream runtime from a local upstream checkout: verify every file against `runtime_source_manifest.json` and write `runtime/upstream` |
| `figures/*.py` | Figures 1–4 and the shared style module `_figstyle_kernel.py` |
| `evidence/` | Every record the analyses and figures read: per-seed held-out evaluations, per-partition results and split protocols, ablation results, literature coding tables, and `evidence/DEPOSIT_MANIFEST.json`, a size and SHA-256 inventory of the whole directory |
| `polymergnn_compatibility.patch` | Compatibility patch against upstream PolymerGNN at the pinned commit |
| `runtime_source_manifest.json` | Path, Git blob hash, SHA-256 and size of the 69 unmodified upstream files the evaluation runtime starts from |
| `runtime_compatible_manifest.json` | The same records for the patched runtime tree |

## Records

`evidence/` holds the study-generated records. The twenty-partition experiment is archived as two JSON Lines
tables in `evidence/multisplit/`. Each line of `runs.jsonl.gz` is one training run, named by its `task`, its
`arm` and its `cell` (`k<partition>_s<seed>`), and carrying the validation loss at the restored checkpoint
(`val_loss`) and the target standardization statistics (`aux`). There are 1200 lines: three tasks, two arms,
twenty partitions and ten seeds, and each task-arm pair has the same twenty partitions. Each line of
`splits.jsonl.gz` is one task-arm-partition `cell`, with the
training, validation and test identifier lists that partition used, the re-split algorithm, the grouping key
and the seed. All ten seeds of a cell share one identifier set, so the table has 120 lines rather than 1200.

`scripts/analyze_multisplit.py` reads both tables and recomputes the split-variance and seed-variance decomposition
reported in Fig. 2. The 296 compounds are divided into a held-out test list of 45 and a development set of 251,
which each partition splits in turn into 207 training and 44 validation compounds; the test identifiers take no
part in training or model selection.

`evidence/r8_sealed_test_eval/` holds the held-out test results as per-task summaries plus a
byte-exact compressed copy of the raw evaluation record (`records_xz_b64.json`, base64-encoded xz, with its
size and SHA-256). The held-out partition was evaluated once, so that record is the primary evidence for the
reported test statistics and cannot be regenerated.

`evidence/r11_literature/` holds the literature-survey records: the corpus retrieved from the indexing services
(`corpus_raw.json`), the per-paper bibliographic metadata and coding decisions (`extraction_nofulltext.json`,
`extraction_v2_nofulltext.json`), the final coding table used for the reported proportions (`coding_table.json`),
and the identifier lists for the full corpus and the polymer subset. The retrieved full texts are not
distributed; the coding table records the bibliographic fields and the assigned category for each paper.

`evidence/DEPOSIT_MANIFEST.json` lists every file under `evidence/` with its size and SHA-256 so that the
archived records can be verified without the dataset or the trained checkpoints. `evidence/multisplit/code_manifest.json`
records the commit of the evaluation harness that produced each task-arm block of runs.

## Running the analyses

Run from the repository root. The analyses write derived summaries into `evidence/`, so run them in a
working copy rather than in the archived records you want to keep unchanged.

```
python3 scripts/analyze_r8.py
python3 scripts/analyze_multisplit.py
```

Several statistics modules execute their analysis on import. `scripts/analyze_round2.py` and
`scripts/analyze_r13_val_test.py` write a summary file into `evidence/` when they run, so import them only to
run them.

Model training is not part of this package, so `scripts/evaluate_smoke.py` cannot run here: it calls the
training entry point. The evaluation harness scores the checkpoints saved by the training runs, and each
checkpoint's own metadata carries the standardization statistics those runs fit on their training split.
The checkpoints are not archived either. The analyzer scripts, by contrast, read only the records in
`evidence/` and reproduce the reported statistics from them alone.

Required packages: Python 3, NumPy and SciPy for the analyses, Matplotlib for the figures. The two
`diagnose_r8_joint_baseline*.py` scripts additionally import TaskGNN from the patched upstream tree and
therefore need a `runtime/compatible` checkout and PyTorch Geometric; they read the archived per-sample
records and can be skipped when only the reported statistics are being checked.

Sign conventions. `scripts/analyze_r8.py` reports improvement as `(baseline - selected) / baseline`.
The manuscript reports test change as `(selected - baseline) / baseline * 100`, where a positive value
means a larger held-out loss. The R13 mean, median and leave-one-seed-out ranges follow the manuscript
convention; the leave-one-seed-out range is a sensitivity range, not a confidence interval. Task losses
are not directly comparable across property targets.

## Figures

```
python3 figures/fig1_protocol_progression.py
```

Each of `figures/fig1_protocol_progression.py` to `figures/fig4_joint_baseline_diagnosis.py` reads its
inputs from `evidence/` at the repository root, so the layout above has to be preserved. Each script
writes a PDF and a PNG beside itself.

## Rebuilding the upstream runtime

```
git clone https://github.com/owencqueen/PolymerGNN.git runtime/upstream
git -C runtime/upstream checkout c0345c1476f137721039c3d7d8e6772d93075794
cp -r runtime/upstream runtime/compatible
git -C runtime/compatible apply ../../polymergnn_compatibility.patch
```

The patch changes three upstream modules (`ablation_models/iv_mono.py`, `joint_test.py`, `tg.py`) so
that they run on the PyTorch Geometric version used for these experiments. It wraps `SAGPooling` so
that the pooling layers inside `Sequential` receive the batch assignment as a positional argument, it
initializes the joint test head and the pretrained-embedding projections that upstream left at the
default initialization, and it assigns the pretrained graph embeddings to the attribute the pooling stack
reads. The patch applies to the pinned commit without fuzz.

Place the clone at `runtime/upstream` and the patched copy at `runtime/compatible`, which is where the
harness scripts look for them. `scripts/materialize_runtime.py` copies the 69 manifest-listed files out of
your `runtime/upstream` checkout, and the block above finishes the tree. That checkout carries the dataset
files (`dataset/pub_data.csv` and `Structures/AG/xyz`), which are listed with hashes in
`runtime_source_manifest.json` and are not redistributed here.

## Citation

Repository: https://github.com/Ymujiang/polymergnn-evaluation-study. Cite the article together with the
archived DOI given in its Data availability statement; the commit published here is the archived record.