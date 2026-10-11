"""Original analyst's data preparation code. Reproduced as-is for review.
This file is intentionally a historical handoff, not a recommended design.
"""
from pathlib import Path
import pandas as pd

FACTOR_COLUMNS = [f'factor_{i:02d}' for i in range(1,21)]


def load_research_frame(data_dir: Path) -> pd.DataFrame:
    m = pd.read_csv(data_dir/'market_daily.csv.gz')
    f = pd.read_csv(data_dir/'factors.csv.gz')
    u = pd.read_csv(data_dir/'universe.csv.gz')
    panel = m.merge(f, on=['date','asset_id'], validate='one_to_one')
    panel = panel.merge(u[['date','asset_id','is_member']], on=['date','asset_id'], validate='one_to_one')
    panel = panel.sort_values(['asset_id','date'])
    # Original project built its label from available prices and treated
    # the vendor factor snapshot as if it were entirely point-in-time.
    g = panel.groupby('asset_id', sort=False)['open']
    panel['forward_5d'] = g.shift(-6)/g.shift(-1)-1.0
    panel = panel.loc[(panel.is_member==True) & panel.forward_5d.notna()].copy()
    return panel
