"""Original research handoff: train and reconstruct the recorded model results.
Dependencies: numpy pandas scikit-learn. Called from package root.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from build_factor import load_research_frame, FACTOR_COLUMNS


def run(root: Path, output: Path):
    df = load_research_frame(root/'data')
    sample = df.sample(n=min(110_000, len(df)), random_state=1021)
    x = sample[FACTOR_COLUMNS].to_numpy(dtype=np.float32)
    y = sample['forward_5d'].to_numpy(dtype=np.float32)
    tr, va = train_test_split(np.arange(len(sample)), test_size=.2, random_state=1021, shuffle=True)
    model = HistGradientBoostingRegressor(max_iter=55, learning_rate=.065,
                  max_leaf_nodes=17, l2_regularization=0.2, random_state=1021)
    model.fit(x[tr],y[tr])
    valpred = model.predict(x[va])
    yv = y[va]
    # The validation is a shuffled row holdout; date coverage is NOT disjoint.
    valid_corr = np.corrcoef(valpred,yv)[0,1]
    rmse=np.sqrt(mean_squared_error(yv,valpred))
    # For continuity with the old analyst's file, predict the complete panel,
    # including training observations. This is not an investable OOS equity curve.
    pred = model.predict(df[FACTOR_COLUMNS].to_numpy(dtype=np.float32))
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'date':df.date,'asset_id':df.asset_id,'prediction':pred.astype('float32')}).to_csv(
        output/'baseline_predictions.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
    pd.DataFrame([
        {'metric':'sampled_rows','value':len(sample)},
        {'metric':'sampled_train_rows','value':len(tr)},
        {'metric':'sampled_valid_rows','value':len(va)},
        {'metric':'random_valid_pearson','value':float(valid_corr)},
        {'metric':'random_valid_rmse','value':float(rmse)},
    ]).to_csv(output/'baseline_results.csv',index=False)
    meta = {'n_features':len(FACTOR_COLUMNS),'factor_columns':FACTOR_COLUMNS,
            'train_test_split':'random 80/20 at row-level',
            'sampled_rows':len(sample),'trained_model':'HistGradientBoostingRegressor',
            'label':'open(t+1) to open(t+6) simple return',
            'full_prediction_count':len(df),'random_seed':1021}
    (output/'baseline_run_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print('Training complete',meta,'valid_corr',valid_corr,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=Path('baseline'))
    a=p.parse_args()
    run(a.root,a.out)
