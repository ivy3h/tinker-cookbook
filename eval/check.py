import pandas as pd

# 读取并查看文件的列名和前几行
df = pd.read_json('results/math500/Qwen3-8B-Base/English-English.jsonl', lines=True)
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)