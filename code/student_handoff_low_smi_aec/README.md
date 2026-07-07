# AEC Low-SMI Reproducible Pipeline

This folder is the student handoff version.

The student only needs the two raw Excel files:

```text
data/g1090.xlsx
data/sdata.xlsx
```

Then run:

```bash
python code/run_from_raw.py
```

The pipeline creates every intermediate file from the raw Excel files:

```text
Stage 1: locked AEC feature search
Stage 2: region-guided CNN branch probabilities
Stage 3: direct-vote AEC score generation
Stage 4: final 20% global-quintile phenotype enrichment analysis
```

The main final output is:

```text
outputs/aec_final_global_quintile_phenotype/01_quintile_vs_quartile_enrichment.csv
```

The primary manuscript result uses 20%:

```text
Clinical+ = top 20% of internal clinical score
AEC-high  = top 20% of AEC score among Clinical+
AEC-low   = bottom 20% of AEC score among Clinical+
```

The 25% analysis is included as sensitivity analysis.

Required Python packages:

```text
numpy
pandas
scipy
scikit-learn
statsmodels
torch
matplotlib
openpyxl
```

For a quick check without model training:

```bash
python code/run_from_raw.py --dry-run
```
