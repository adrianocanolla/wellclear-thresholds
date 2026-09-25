# DO-365 Well Clear thresholds in terminal airspace — analysis code

Code and results for *Data-Driven Validation of DO-365 Detect-and-Avoid Well
Clear Thresholds for UAS in Terminal Airspace Using One Million Encounters*.

The study evaluates the RTCA DO-365 en-route Well Clear thresholds
(DMOD = 4000 ft, ZTHR = 450 ft, TAUMOD = 35 s) against terminal-area
encounters, and compares them with the terminal range recommended by
Vincent et al. (2018). Ground truth is the standard NMAC definition
(500 ft / 100 ft), computed from each encounter's true closest point of
approach.

> **Citation:** Canolla, A. *Data-Driven Validation of DO-365
> Detect-and-Avoid Well Clear Thresholds for UAS in Terminal Airspace
> Using One Million Encounters.* Drones (MDPI), forthcoming.
> 
> Code archive: [![DOI](https://zenodo.org/badge/1370680378.svg)](https://doi.org/10.5281/zenodo.22833837)


---

## Getting the data

The encounter dataset is **not included** — the trajectory archive alone is
18 GB. Download it separately:

**MIT Lincoln Laboratory Terminal Encounter Model (LLTEM) V1.0**, 30 June 2020,
licensed **CC BY 4.0**, available from MIT Lincoln Laboratory:

<https://www.ll.mit.edu/r-d/datasets/unmanned-aircraft-terminal-area-encounters>

| File | Size | Needed? |
|---|---|---|
| `terminal_encounter_info_20200630.csv` | 139 MB | **yes** — encounter metadata, CPA conditions |
| `terminal_encounter_state_data_20200630.zip` | 18 GB | **yes** — per-encounter 1 Hz extended state |
| `terminal_encounters_20200630.dat` | — | no — position-only binary, unused here |

Put both required files in `data/`. The archive is read member by member and is
never extracted.

Dataset questions go to MIT Lincoln Laboratory (contact details are in the dataset's own `README.txt`).

## Running it

```bash
pip install -r requirements.txt
jupyter notebook reproduce_paper.ipynb
```

Run the notebook top to bottom. Section 4 preprocesses 60,000 encounters and
takes roughly an hour; it caches to `results/stacked_cache.npz`, so sections 5
onward re-run in minutes afterwards. Delete that file to force a fresh pass.

The notebook ships with its outputs saved, so every table and figure can be
read without downloading the dataset or running anything.

## What is here

| File | What it does |
|---|---|
| `reproduce_paper.ipynb` | The analysis, in 14 sections numbered to match the manuscript |
| `lltem_preprocessing.py` | Reading the archive, building relative state, aligning to CPA, smoothing |
| `well_clear_analysis.py` | CPA metrics, LoWC evaluation, detection metrics, bootstrap, predicted-CPA accuracy |
| `results/` | Every number the manuscript cites, as JSON and CSV |
| `figures/` | The four figures, as PDF |

### Which output backs which table or figure

| Manuscript | File |
|---|---|
| §5.1 population and sample | `population_summary.json`, `sample_summary.json` |
| Table 1 — baseline vs. Vincent et al. | `named_region_metrics.json` |
| Table 2 — by encounter geometry | `named_region_metrics_by_geometry.json` |
| Table 3 — population-reweighted | `population_reweighted_fpr_region.json` |
| §5.3 false-positive composition | `false_positive_composition.json` |
| Table 4 — smoothing sensitivity | `smoothing_sensitivity.json` |
| Table 5 — predicted-CPA accuracy | `predicted_cpa_accuracy.json`, `predicted_cpa_engaged_only.json` |
| Table 6 — predicted-CPA by stratum | `predicted_cpa_strata.json` |
| §5.3 bootstrap intervals | `bootstrap_named_configs_region.json` |
| §5.7 region accuracy | `region_accuracy.json` |
| §5.8 alert rates and lead times | `operational_feasibility.json` |
| Figure 1 | `figures/methodology_workflow.pdf` |
| Figure 2 | `figures/threshold_tradeoff_heatmap.pdf`, `fpr_vs_taumod.json` |
| Figure 3 | `figures/confusion_matrix.pdf` |
| Figure 4 | `figures/alert_leadtime_distribution.pdf` |

## Reproducibility parameters

Everything below is fixed in code; changing any of it changes the results.

| Parameter | Value |
|---|---|
| Sample | 60,000 encounters, equal quota per geometry class |
| Random seed | 42 |
| Alignment window | ±60 s about the metadata CPA time |
| Resampling | linear interpolation onto a uniform 1 Hz grid |
| Smoothing | Savitzky–Golay, polynomial order 3, window 7 samples |
| Derivatives | analytic, from the same local polynomial |
| Runway origin | 42.4699° N, −71.2874° W, oriented due north |
| Bootstrap | 1,000 percentile replicates, pooled and geometry-stratified |

### A note on reading the state files

The extended-state CSVs carry one header line and a **trailing comma on every
row**, so each row parses into 16 fields rather than 15. Supplying only 15
column names makes pandas absorb the first field as the index and shift every
remaining name one position left. The loader here passes a trailing throwaway name and `skiprows=1` to keep each
column aligned with its contents. See `lltem_preprocessing.load_encounter_pair`.


## License

Code: MIT, see `LICENSE`. The LLTEM dataset is CC BY 4.0 from MIT Lincoln
Laboratory and is not redistributed here.
