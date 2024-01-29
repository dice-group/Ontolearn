import pandas as pd

pd.set_option('display.max_columns', None)
dataset = "Carcinogenesis"
directory = f"{dataset}BenchmarkResults"

df1 = pd.read_csv(f"{directory}/{dataset.lower()}_results1.csv")
df2 = pd.read_csv(f"{directory}/{dataset.lower()}_results2.csv")
df3 = pd.read_csv(f"{directory}/{dataset.lower()}_results3.csv")
df4 = pd.read_csv(f"{directory}/{dataset.lower()}_results4.csv")
df5 = pd.read_csv(f"{directory}/{dataset.lower()}_results5.csv")
dfs = pd.concat([df1, df2, df3, df4, df5]).groupby(by="LP", as_index=False).mean()

# print(dfs.mean(numeric_only=True))
print(dfs.to_latex(index=False, formatters={"name": str.upper}, float_format="{:.3f}".format))

# print(dfs.to_markdown(index=False, floatfmt=".3f"))
