# %%
# import libraries
from interpret_community import TabularExplainer
import pandas as pd
import joblib
import argparse
from sklearn import datasets
from azureml.core.dataset import Dataset
import os

# %%
# Parse the arguments
parser = argparse.ArgumentParser(
    description="getting inputs from the pipeline setup")
parser.add_argument("--model_data")
parser.add_argument("--linear_gold")
parser.add_argument("--shap_tables")

args = parser.parse_args()


# %%
# Reading the file from the input.
# inputpath = "../../data/linear_gold.csv"
inputpath = os.path.join(args.linear_gold, "linear_gold.csv")
df = pd.read_csv(inputpath)
print(df.head())


model = joblib.load(args.model_data)
# model = joblib.load("../../models/model.pkl")
print(model)

# %%

# get SHAP values
features_df = df.drop('price', axis=1)
explainer = TabularExplainer(model, features_df, features=features_df.columns)
shap_values = explainer.explain_global(features_df).local_importance_values


df_shap = pd.DataFrame(shap_values, columns=features_df.columns)
print(df_shap.head())

# %%
# Save output
os.makedirs(args.shap_tables, exist_ok=True)
outputpath = os.path.join(args.shap_tables, "linear_gold_shap_1.csv")
df_shap.to_csv(outputpath)
