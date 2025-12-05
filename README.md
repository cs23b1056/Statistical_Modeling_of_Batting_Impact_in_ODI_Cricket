# 🏏 ODI Cricket Analysis

This repository packages the final ODI batting analysis contained in `ODI_analysis_v1.ipynb`. The notebook consolidates data preparation, exploratory insights, clustering, phase analysis, and a match-outcome classifier that reports accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

## ⚡ Quick Start
- `pip install -r requirements.txt`
- Launch Jupyter and open `ODI_analysis_v1.ipynb`
- Execute the notebook sequentially to reproduce metrics and plots

## 📂 Project Contents
- `ODI_analysis_v1.ipynb` – primary end-to-end analysis
- `odi_dataset.csv` – source ball-by-ball dataset used by the notebook
- `figures/` – final PNG exports referenced in documentation
  - `1_early_momentum.png`
  - `2_wicket_impact.png`
  - `3_chase_vs_defend.png`
  - `4_target_impact.png`
- `reports/` – supporting tables for presentation/reporting
  - `cluster_player_table_with_names.csv`
  - `phase_sr_thresholds.csv`
  - `player_id_map.csv`
  - `regression_summary.txt`
- `src/` – helper modules for reusable preprocessing, clustering, visualization, and presentation pipelines
- `requirements.txt` – Python dependencies for notebook execution

## 📊 Notebook Highlights
- Cleans and aggregates 1.36M ball-level records into batsman-innings features
- Compares batting eras, strike-rate bands, and phase-specific win probabilities
- Identifies batting style clusters with silhouette-based KMeans selection
- Evaluates outcome prediction using logistic regression with hold-out metrics (accuracy, precision, recall, F1, ROC-AUC)
- Generates heatmaps and trend plots saved to `figures/`

## ✅ Outputs
- Visual summaries captured in `figures/`
- Cluster memberships and phase thresholds in `reports/`
- Classification metrics displayed directly in the notebook

## 🔄 Reproducibility
- Random seeds fixed where applicable (`random_state=42`)
- Notebook can be re-run end-to-end with the provided dataset and requirements
- Source modules in `src/` mirror the logic for scripted execution if needed

For any future extensions, continue building from `ODI_analysis_v1.ipynb` and export refreshed visuals or tables into the existing `figures/` and `reports/` directories.
