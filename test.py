import pandas as pd

df = pd.read_parquet("embeddings/thrangu_rinpoche_embeddings.parquet")
print(df.columns)
