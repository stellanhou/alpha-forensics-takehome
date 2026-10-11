"""Original analyst's simplified cross-sectional paper backtest.
It computes contemporaneous horizon spreads, not realizable daily cash P&L.
Provided for handoff review, not as a template to follow.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from build_factor import load_research_frame


def run(root: Path, output:Path):
    research=load_research_frame(root/'data')
    preds=pd.read_csv(output/'baseline_predictions.csv.gz')
    df=research[['date','asset_id','forward_5d']].merge(preds,on=['date','asset_id'],validate='one_to_one')
    daily=[]
    for date,sub in df.groupby('date',sort=True):
        if len(sub)<60:continue
        ql=sub.prediction.quantile(.1)
        qh=sub.prediction.quantile(.9)
        l=sub.loc[sub.prediction>=qh,'forward_5d'].mean()
        s=sub.loc[sub.prediction<=ql,'forward_5d'].mean()
        daily.append((date,float(l-s),float(l)))
    daily=pd.DataFrame(daily,columns=['date','long_short_5d_spread','long_5d_return'])
    daily.to_csv(output/'baseline_backtest.csv',index=False)
    spread=daily.long_short_5d_spread.to_numpy()
    # Approximate proxy based on heavily overlapping forward windows.
    # It ignores transaction fees, spread, capacity, and execution constraints.
    proxy_sharpe=float(spread.mean()/(spread.std()+1e-12)*np.sqrt(252))
    rows=[{'metric':'full_panel_gross_proxy_sharpe','value':proxy_sharpe},
          {'metric':'full_panel_mean_forward_5d_spread','value':float(spread.mean())},
          {'metric':'backtest_n_dates','value':len(daily)}]
    current=pd.read_csv(output/'baseline_results.csv')
    pd.concat([current,pd.DataFrame(rows)],ignore_index=True).to_csv(output/'baseline_results.csv',index=False)
    print('Paper spread proxy',rows,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=Path('baseline'))
    a=p.parse_args()
    run(a.root,a.out)
