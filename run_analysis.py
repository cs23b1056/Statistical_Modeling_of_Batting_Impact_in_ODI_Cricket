"""Run regression, phase-wise SR contributions, and cluster labeling for ODI dataset.

Outputs (saved in reports/):
- regression_summary.txt
- phase_sr_thresholds.csv
- cluster_player_table.csv

This script is intended to be run from the repository root.
"""
import os
import warnings
# Do not globally silence warnings. We'll address specific pandas deprecation by
# using explicit groupby parameters where needed. Keep default warning behavior
# so important messages from libraries are visible during runs.

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, 'odi_dataset.csv')
REPORTS = os.path.join(ROOT, 'reports')
os.makedirs(REPORTS, exist_ok=True)

print('Loading data...')
df = pd.read_csv(DATA_PATH, low_memory=False)
print('Rows:', len(df))

# Map player id (`p_bat`) to human-readable name if available in the dataset
# many datasets include a `bat` column with the batter's name; create a small
# lookup table and save it for downstream joins and reporting.
if 'bat' in df.columns:
    pbat_map = df[['p_bat', 'bat']].drop_duplicates().rename(columns={'bat': 'player_name'})
    # keep one-to-one mapping by taking first occurrence per id
    pbat_map = pbat_map.groupby('p_bat', observed=False).first().reset_index()
    # save mapping for reference
    pbat_map.to_csv(os.path.join(REPORTS, 'player_id_map.csv'), index=False)
    print('Player id -> name mapping saved to reports/player_id_map.csv')
else:
    pbat_map = pd.DataFrame(columns=['p_bat', 'player_name'])

# Preprocess similarly to notebook
print('Preprocessing and aggregating to batsman-innings...')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['era'] = df['year'].apply(lambda x: 'pre-2005' if x < 2005 else 'modern')

# aggregate
batsman_inns = df.groupby(['p_match', 'inns', 'p_bat']).agg({
    'batruns': 'sum',
    'ballfaced': 'sum',
    'team_bat': 'first',
    'winner': 'first',
    'year': 'first',
    'era': 'first'
}).reset_index()

batsman_inns['strike_rate'] = batsman_inns['batruns'] / batsman_inns['ballfaced'] * 100
batsman_inns['innings_length'] = batsman_inns['ballfaced']
batsman_inns['result'] = (batsman_inns['team_bat'] == batsman_inns['winner']).astype(int)

# filter
b = batsman_inns[(batsman_inns['ballfaced'] > 0) & (batsman_inns['strike_rate'].between(0,200))].copy()
print('Aggregated innings:', len(b))

# Feature engineering for model
b['sr_powerplay'] = np.nan
b['sr_middle'] = np.nan
b['sr_death'] = np.nan
# Note: ball-by-ball SR per phase would be ideal; we don't have per-phase per-batsman aggregated here. Skip phase-level per-batsman SR unless detailed data available.

# We'll build a model using strike_rate and innings_length (and era) to predict result.
features = ['strike_rate', 'innings_length']
X = b[features].fillna(0)
y = b['result']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale
scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)
Xte = scaler.transform(X_test)

# Logistic regression
print('Fitting logistic regression...')
clf = LogisticRegression(max_iter=200)
clf.fit(Xtr, y_train)
probs = clf.predict_proba(Xte)[:,1]
preds = clf.predict(Xte)

# Compute all metrics for Logistic Regression
lr_auc = roc_auc_score(y_test, probs)
lr_acc = accuracy_score(y_test, preds)
lr_prec = precision_score(y_test, preds, zero_division=0)
lr_rec = recall_score(y_test, preds, zero_division=0)
lr_f1 = f1_score(y_test, preds, zero_division=0)
lr_mcc = matthews_corrcoef(y_test, preds)
lr_fpr, lr_tpr, _ = roc_curve(y_test, probs)
lr_roc_auc = auc(lr_fpr, lr_tpr)
lr_cm = confusion_matrix(y_test, preds)
lr_tn, lr_fp, lr_fn, lr_tp = lr_cm.ravel()
lr_specificity = lr_tn / (lr_tn + lr_fp) if (lr_tn + lr_fp) > 0 else 0
lr_sensitivity = lr_rec  # sensitivity = recall
lr_coefs = dict(zip(features, clf.coef_[0]))
lr_intercept = clf.intercept_[0]

# Random Forest for feature importance and metrics
print('Fitting RandomForestClassifier...')
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:,1]
rf_preds = rf.predict(X_test)
rf_importances = dict(zip(features, rf.feature_importances_))

# Compute all metrics for Random Forest
rf_auc = roc_auc_score(y_test, rf_probs)
rf_acc = accuracy_score(y_test, rf_preds)
rf_prec = precision_score(y_test, rf_preds, zero_division=0)
rf_rec = recall_score(y_test, rf_preds, zero_division=0)
rf_f1 = f1_score(y_test, rf_preds, zero_division=0)
rf_mcc = matthews_corrcoef(y_test, rf_preds)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
rf_roc_auc = auc(rf_fpr, rf_tpr)
rf_cm = confusion_matrix(y_test, rf_preds)
rf_tn, rf_fp, rf_fn, rf_tp = rf_cm.ravel()
rf_specificity = rf_tn / (rf_tn + rf_fp) if (rf_tn + rf_fp) > 0 else 0
rf_sensitivity = rf_rec

# Save comprehensive regression summary
reg_file = os.path.join(REPORTS, 'regression_summary.txt')
with open(reg_file, 'w', encoding='utf8') as f:
    f.write('='*80 + '\n')
    f.write('ODI BATTING ANALYSIS: REGRESSION & CLASSIFICATION METRICS\n')
    f.write('='*80 + '\n\n')
    
    f.write('LOGISTIC REGRESSION (Standardized Features: Strike Rate, Innings Length)\n')
    f.write('-'*80 + '\n')
    f.write('Model Hyperparameters: max_iter=200, solver=lbfgs\n\n')
    
    f.write('CLASSIFICATION METRICS:\n')
    f.write(f'  Accuracy:        {lr_acc:.4f}\n')
    f.write(f'  Precision:       {lr_prec:.4f}  (of predicted wins, how many were correct?)\n')
    f.write(f'  Recall:          {lr_rec:.4f}  (of actual wins, how many did we find?)\n')
    f.write(f'  Sensitivity:     {lr_sensitivity:.4f}  (true positive rate)\n')
    f.write(f'  Specificity:     {lr_specificity:.4f}  (true negative rate)\n')
    f.write(f'  F1-Score:        {lr_f1:.4f}  (harmonic mean of precision & recall)\n')
    f.write(f'  Matthews Corr.:  {lr_mcc:.4f}  (balanced measure for imbalanced classes)\n')
    f.write(f'  AUC (ROC):       {lr_auc:.4f}  (probability model is well-calibrated)\n')
    f.write(f'  ROC-AUC:         {lr_roc_auc:.4f}  (area under ROC curve)\n\n')
    
    f.write('CONFUSION MATRIX:\n')
    f.write(f'  True Negatives:  {lr_tn}\n')
    f.write(f'  False Positives: {lr_fp}\n')
    f.write(f'  False Negatives: {lr_fn}\n')
    f.write(f'  True Positives:  {lr_tp}\n\n')
    
    f.write('MODEL COEFFICIENTS (Feature Weights):\n')
    for k, v in lr_coefs.items():
        f.write(f'  {k:20s}: {v:8.6f}\n')
    f.write(f'  Intercept        : {lr_intercept:8.6f}\n\n')
    
    f.write('\n' + '='*80 + '\n')
    f.write('RANDOM FOREST CLASSIFIER (100 estimators)\n')
    f.write('-'*80 + '\n')
    f.write('Model Hyperparameters: n_estimators=100, random_state=42, n_jobs=2\n\n')
    
    f.write('CLASSIFICATION METRICS:\n')
    f.write(f'  Accuracy:        {rf_acc:.4f}\n')
    f.write(f'  Precision:       {rf_prec:.4f}  (of predicted wins, how many were correct?)\n')
    f.write(f'  Recall:          {rf_rec:.4f}  (of actual wins, how many did we find?)\n')
    f.write(f'  Sensitivity:     {rf_sensitivity:.4f}  (true positive rate)\n')
    f.write(f'  Specificity:     {rf_specificity:.4f}  (true negative rate)\n')
    f.write(f'  F1-Score:        {rf_f1:.4f}  (harmonic mean of precision & recall)\n')
    f.write(f'  Matthews Corr.:  {rf_mcc:.4f}  (balanced measure for imbalanced classes)\n')
    f.write(f'  AUC (ROC):       {rf_auc:.4f}  (probability model is well-calibrated)\n')
    f.write(f'  ROC-AUC:         {rf_roc_auc:.4f}  (area under ROC curve)\n\n')
    
    f.write('CONFUSION MATRIX:\n')
    f.write(f'  True Negatives:  {rf_tn}\n')
    f.write(f'  False Positives: {rf_fp}\n')
    f.write(f'  False Negatives: {rf_fn}\n')
    f.write(f'  True Positives:  {rf_tp}\n\n')
    
    f.write('FEATURE IMPORTANCES:\n')
    for k, v in rf_importances.items():
        f.write(f'  {k:20s}: {v:8.6f}\n\n')
    
    f.write('='*80 + '\n')
    f.write('METRIC DEFINITIONS\n')
    f.write('='*80 + '\n')
    f.write('Accuracy:       (TP + TN) / (TP + TN + FP + FN) - overall correctness\n')
    f.write('Precision:      TP / (TP + FP) - of predicted positives, what % are correct?\n')
    f.write('Recall:         TP / (TP + FN) - of actual positives, what % did we find?\n')
    f.write('Sensitivity:    TP / (TP + FN) - same as recall (true positive rate)\n')
    f.write('Specificity:    TN / (TN + FP) - true negative rate\n')
    f.write('F1-Score:       2 * (Precision * Recall) / (Precision + Recall) - balance metric\n')
    f.write('Matthews Corr.: balanced measure; works well with imbalanced datasets\n')
    f.write('AUC:            Area Under ROC Curve; 0.5=random, 1.0=perfect classification\n')
    f.write('MCC:            Matthews Correlation Coefficient; range [-1, 1]; 1=perfect\n')
    f.write('\nWhere: TP=True Positives, TN=True Negatives, FP=False Positives, FN=False Negatives\n')

print('Regression results saved to', reg_file)

# Phase-wise SR contributions and thresholds
# We need to compute SR per phase per batsman; fallback: compute phase contributions approximately using ball-level data
print('Computing phase-wise SR contributions (using ball-level data aggregation)...')
# We'll derive per-batsman phase SR by aggregating original ball-level df using 'over' to assign phases.

df_ball = df.copy()
# assign phase
bins = [0,10,40,50]
labels = ['powerplay','middle','death']
df_ball['phase'] = pd.cut(df_ball['over'], bins=bins, labels=labels)
# Some rows may have NaN phase; drop them
phase_agg = df_ball.dropna(subset=['phase']).groupby(['p_match','inns','p_bat','phase'], observed=False).agg({'cur_bat_runs':'sum','cur_bat_bf':'sum'}).reset_index()
phase_agg = phase_agg.rename(columns={'cur_bat_runs':'runs_phase','cur_bat_bf':'bf_phase'})
phase_agg['sr_phase'] = phase_agg['runs_phase'] / phase_agg['bf_phase'] * 100
# merge total result and era
meta = batsman_inns[['p_match','inns','p_bat','result','era']]
phase_agg = phase_agg.merge(meta, on=['p_match','inns','p_bat'], how='left')

# For each phase, compute win probability by SR bin and detect threshold where win prob increases notably
phase_thresholds = []
for ph in labels:
    ph_df = phase_agg[phase_agg['phase']==ph].dropna(subset=['sr_phase','bf_phase'])
    if ph_df.empty:
        continue
    # remove zero bf
    ph_df = ph_df[ph_df['bf_phase']>0]
    # bin SR
    ph_df['sr_bin'] = pd.cut(ph_df['sr_phase'], bins=[0,50,70,85,100,120,200], labels=['0-50','50-70','70-85','85-100','100-120','120+'])
    win_rates = ph_df.groupby('sr_bin', observed=False)['result'].agg(['count','mean']).reset_index()
    # choose threshold where mean increases above previous by >0.05
    win_rates['mean_shift'] = win_rates['mean'].diff()
    candidate = win_rates[win_rates['mean_shift']>0.05]
    threshold = candidate['sr_bin'].iloc[0] if not candidate.empty else win_rates.iloc[win_rates['mean'].idxmax()]['sr_bin']
    phase_thresholds.append({'phase':ph, 'threshold_bin':str(threshold), 'win_rates':win_rates})

# Save thresholds summary
th_list = []
for item in phase_thresholds:
    ph = item['phase']
    thr = item['threshold_bin']
    wr = item['win_rates']
    for _,row in wr.iterrows():
        th_list.append({'phase':ph, 'sr_bin':row['sr_bin'], 'count':row['count'], 'win_rate':row['mean']})

th_df = pd.DataFrame(th_list)
th_df.to_csv(os.path.join(REPORTS, 'phase_sr_thresholds.csv'), index=False)
print('Phase SR thresholds saved to reports/phase_sr_thresholds.csv')

# Clustering and labeling players
print('Clustering batsman innings and labeling players...')
cluster_features = b[['strike_rate','innings_length']].fillna(0)
sc = StandardScaler().fit(cluster_features)
Xc = sc.transform(cluster_features)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(Xc)
b['cluster'] = labels

centroids = pd.DataFrame(sc.inverse_transform(kmeans.cluster_centers_), columns=['strike_rate','innings_length'])
centroids['cluster'] = centroids.index
print('Cluster centroids:')
print(centroids)

# Map clusters to human labels based on centroid rules
# We'll identify cluster with highest innings_length and low SR as 'Anchor', high SR low length as 'Finisher/Attacker', high both as 'Aggressive Anchor'
centroids['label'] = 'unknown'
# heuristics
sr_means = centroids['strike_rate']
len_means = centroids['innings_length']
for idx,row in centroids.iterrows():
    if row['strike_rate']>=100 and row['innings_length']<40:
        centroids.loc[idx,'label']='Attacker'
    elif row['strike_rate']<90 and row['innings_length']>=40:
        centroids.loc[idx,'label']='Anchor'
    elif row['strike_rate']>=90 and row['innings_length']>=40:
        centroids.loc[idx,'label']='Hybrid'
    else:
        centroids.loc[idx,'label']='Accumulator'

# invert mapping cluster->label
cluster_label_map = centroids.set_index('cluster')['label'].to_dict()

b['style'] = b['cluster'].map(cluster_label_map)

# Produce table of typical players per style (need enough innings)
player_stats = b.groupby(['p_bat','style']).agg({'batruns':'sum','innings_length':'mean','strike_rate':'mean','p_match':'count'}).rename(columns={'p_match':'innings_count'}).reset_index()
# filter players with >= 30 innings
player_stats_filt = player_stats[player_stats['innings_count']>=30]
# pick top players by innings_count per style
out_rows = []
for style,grp in player_stats_filt.groupby('style'):
    top = grp.sort_values('innings_count', ascending=False).head(20)
    top['style'] = style
    out_rows.append(top)
if out_rows:
    players_table = pd.concat(out_rows)
else:
    players_table = player_stats_filt

players_table.to_csv(os.path.join(REPORTS,'cluster_player_table.csv'), index=False)
print('Cluster player table saved to reports/cluster_player_table.csv')

# Merge human-readable player names into the table if mapping exists and save
if not pbat_map.empty:
    players_with_names = players_table.merge(pbat_map, on='p_bat', how='left')
    # reorder columns to show player name first
    cols = ['p_bat', 'player_name'] + [c for c in players_with_names.columns if c not in ['p_bat', 'player_name']]
    players_with_names = players_with_names[cols]
    players_with_names.to_csv(os.path.join(REPORTS, 'cluster_player_table_with_names.csv'), index=False)
    print('Cluster player table with names saved to reports/cluster_player_table_with_names.csv')

print('Done.')
