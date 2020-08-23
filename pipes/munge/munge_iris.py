# %%
# import libraries
import pandas as pd
import argparse
from sklearn import datasets
import os

# %%
# Parse the arguments
parser = (argparse.ArgumentParser(
    description="getting inputs from the pipeline setup"))
parser.add_argument('--output_dir', dest="output_dir")
parser.add_argument('--input_dir', dest="input_dir")
args = parser.parse_args()


# %%
# Reading the file from the input.

df = pd.read_csv(os.path.join(args.input_dir, 'iris.csv'), index_col=0)


# %%
# Saving the output file.
print("rows total: ", len(df))
outpath = os.path.join(args.output_dir, 'iris.csv')
df.to_csv('iris.csv')
print("file saved to:", outpath)
