# %%
# The munge step's only purpose is to pick up the data and put it down again.
# Your code would have some kind of transformation step here.

# import libraries
import pandas as pd
import argparse
from sklearn import datasets
import os

# %%
# Parse the arguments
parser = argparse.ArgumentParser(
    description="getting inputs from the pipeline setup")
parser.add_argument("--output_dir", dest="output_dir")
parser.add_argument("--input_dir", dest="input_dir")
args = parser.parse_args()


# %%
# Reading the file from the input.
inputpath = os.path.join(args.input_dir, "initial_data_step.csv")
df = pd.read_csv(inputpath)
print(df.head())
print(df.dtypes)

# %%
# Saving the output file.
os.makedirs(args.output_dir, exist_ok=True)

print("rows total: ", len(df))
outputpath = os.path.join(args.output_dir, "linear_gold.csv")
df.to_csv(outputpath, index=False)
print("file saved to:", outputpath)
