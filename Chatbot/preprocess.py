import os
import pandas as pd
df = pd.read_parquet("test-00000-of-00001.parquet")

df.to_csv('algebra.csv', index=False)

#print(os.path.dirname(__file__))
