# %% Load Packages

import os
import sys
import datetime
import yaml
from datetime import date

from azureml.core import VERSION, Workspace, Experiment
from azureml.core.compute import ComputeTarget
from azureml.core.authentication import InteractiveLoginAuthentication

sys.path.append(os.getcwd())
print("Azure ML SDK Version: ", VERSION)

# %%
# loading project params
params = yaml.safe_load(open('settings.yaml'))


# %%
# Connect to ML Service
interactive_auth = InteractiveLoginAuthentication(
    tenant_id=params.get('tenant_id'))

ws = Workspace.get(name=params.get("workspace_name"),
                   subscription_id=params.get('subscriptionId'),
                   resource_group=params.get('resource_group'),
                   auth=interactive_auth)
print(f"Found workspace {ws.name} at location {ws.location}")
ws.set_default_datastore("factset")
exp = Experiment(ws, params.get("expermient_name"))
compute_target = ComputeTarget(workspace=ws, name=params.get("compute_name"))

# %%
# Printing file systems
print("these are the connected file systems available now")
datastores = ws.datastores
for name, datastore in datastores.items():
    print(name+" " + datastore.datastore_type)


# %%
# Getting some global variables that often come up.
# These are common folder structures in flat file storage.
today = date.today()
d1 = today.strftime("%Y-%m-%d")
year = d1.split("-")[0]
month = d1.split("-")[1]

# If you want to pass global parameters into a script step, without having to have a bunch of little yaml files
# this will allow you to push your single settings.yaml file into each pipe step identified.


def save_params(steps):
    for step in steps:
        save_path = f'pipes/{step}/settings.yaml'
        print(f'params moved to :{save_path}')
        with open(save_path, 'w') as file:
            yaml.dump(params, file)
    with open('notebooks/settings.yaml', 'w') as file:
        yaml.dump(params, file)
