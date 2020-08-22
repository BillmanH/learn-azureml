# %%
# import libraries
import pandas as pd
import argparse
from sklearn import datasets

# %%
# Parse the arguments
parser = (argparse.ArgumentParser(
    description="Starts the data transformation"))
parser.add_argument('--factset_dir', dest="factset_dir",
                    default="pipes/data/",)

# %%
# do raw data extraction
iris = datasets.load_iris()
print(iris.target_names)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[x] for x in iris.target]
print(df.head())
print("rows total: ", len(df))

# %%
# Saving the output file.
print("rows total: ", len(df))
