"""Optional conversion from compressed CSV to Parquet when pyarrow is installed.
Usage: python tools/optional_csv_to_parquet.py
"""
from pathlib import Path
import pandas as pd
D=Path(__file__).resolve().parents[1]/'data'
for csv in sorted(D.glob('*.csv.gz')):
    dst=D/(csv.name.removesuffix('.csv.gz')+'.parquet')
    data=pd.read_csv(csv)
    data.to_parquet(dst,engine='pyarrow',index=False,compression='zstd')
    print(dst,len(data))
