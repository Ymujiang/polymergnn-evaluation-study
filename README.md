# PolymerGNN evaluation study: analysis code

Companion code for “A development-set improvement is not evidence of generalisation: variance decomposition and a sealed test evaluation for polymer property prediction”.

## Scope of this initial release

Six independently written statistical analysis scripts are included. This is a **code-only, partial reproducibility release**, not a complete training package. It does not contain model inference, upstream model code, datasets, checkpoints, the manuscript, or result records. No test evaluation was rerun for this release.

Upstream: https://github.com/owencqueen/PolymerGNN at commit `c0345c1476f137721039c3d7d8e6772d93075794`. No upstream licence was identified; upstream and modified upstream source are therefore not redistributed. No open-source licence has yet been selected for this release; public visibility does not grant an open-source licence.

## Scripts and inputs

Run from the repository root, only in a separate working copy with the required result records supplied. Scripts write derived JSON summaries and must not be run against an immutable evidence archive. Some execute on import.

| Script | Required input location |
|---|---|
| analyze_multiseed.py | evidence/multiseed/ |
| analyze_round2.py | development-run result records (see script glob patterns) |
| analyze_multisplit.py | evidence/multisplit/ |
| analyze_r8.py | evidence/r8_sealed_test_eval/{IV,Tg,joint}.json |
| analyze_r9.py | evidence/multisplit/joint/best/ and evidence/r9_ablation/ |
| analyze_r13_val_test.py | evidence/r8_sealed_test_eval/{IV,Tg,joint}.json |

Python 3, NumPy and SciPy are required. This initial release is syntax-checked only; it has not been end-to-end tested as a standalone package. Input result records and figure scripts are not included, so the repository alone cannot regenerate manuscript results.

The R8 analysis uses improvement=(baseline-selected)/baseline; the manuscript test-change convention uses the opposite sign. Positive manuscript test change means worse performance. Task losses are not directly comparable.

## Citation

Repository: https://github.com/Ymujiang/polymergnn-evaluation-study

Cite the repository URL and the exact commit used. No archival DOI is assigned.
