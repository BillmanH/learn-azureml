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
args = parser.parse_args()

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
if not os.path.exists('args.output_dir'):
    os.makedirs('args.output_dir')

outpath = os.path.join(args.output_dir, 'iris.csv')
df.to_csv('iris.csv')
print("file saved to:", outpath)
