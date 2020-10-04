# %%
# import libraries
import pandas as pd
import yaml
import argparse
from sklearn import datasets
import os

# %%
# Parse the arguments
parser = argparse.ArgumentParser(
    description="getting inputs from the pipeline setup")
parser.add_argument("--output_dir", dest="output_dir")
args = parser.parse_args()

# %%
# do raw data extraction (this is the mid-section that will change in your code)
boston = datasets.load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["price"] = boston.target
print(df.head())
print("rows total: ", len(df))

# %%
# make the directory if it doesn't exist.
os.makedirs(args.output_dir, exist_ok=True)

# Saving the output file.
outpath = os.path.join(args.output_dir, "initial_data_step.csv")
df.to_csv(outpath, index=False)
print("file saved to:", outpath)
