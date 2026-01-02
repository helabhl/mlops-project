import pandas as pd

df = pd.read_csv("data/iris.csv")

df = df.drop_duplicates()

df["petal_area"] = df["petal length (cm)"] * df["petal width (cm)"]

df.to_csv("data/iris.csv", index=False)

