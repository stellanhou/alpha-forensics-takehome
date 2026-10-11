"""Data-only smoke test. It neither evaluates nor solves the research problem."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
files=['market_daily','factors','risk_exposure','universe','trading_constraints','factor_availability']
for name in files:
    path=ROOT/'data'/f'{name}.csv.gz'
    sample=pd.read_csv(path,nrows=5)
    assert {'date','asset_id'}<=set(sample.columns)
    print(f'{name}: OK, example columns={len(sample.columns)}')
print('Data reader smoke test passed.')
