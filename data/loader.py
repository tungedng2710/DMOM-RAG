import pandas as pd
from pathlib import Path

def save_df(df, path):
    ext = Path(path).suffix.lower()
    if ext in {'.csv', '.tsv'}:
        sep = '\t' if ext == '.tsv' else ','
        df.to_csv(path, index=False, sep=sep)
    elif ext in {'.parquet'}:
        df.to_parquet(path, index=False)  # pip install pyarrow
    elif ext in {'.json'}:
        df.to_json(path, orient='records')
    elif ext in {'.pkl'}:
        df.to_pickle(path)
    elif ext in {'.xlsx'}:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f'Unsupported extension: {ext}')


df = pd.read_csv("hf://datasets/tungedng2710/Dmom_dataset/train.csv")
save_df(df, "dmom_data.csv")