# %%
# import libraries
import pandas as pd
import argparse
from sklearn import datasets
from azureml.core.dataset import Dataset
import os

# %%
# Parse the arguments
parser = argparse.ArgumentParser(
    description="getting inputs from the pipeline setup")
parser.add_argument("--output_dir", dest="output_dir")
parser.add_argument("--register_name", dest="register_name")
args = parser.parse_args()


# %%
# Reading the file from the input.
inputpath = os.path.join(args.input_dir, "iris.csv")
df = pd.read_csv(inputpath, index_col=0)
print(df.head())


# %%
dataset = Dataset.File.from_files(path=inputpath)


# %%
# Saving the output file.
registered_dataset = dataset.register(workspace=ws,
                                      name='iris_gold',
                                      description=description_text)
