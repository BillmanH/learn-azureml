# %%
# Load Libraries
from azureml.core.runconfig import (CondaDependencies, RunConfiguration)
from azureml.data.data_reference import DataReference

# my custom libraries
import config as f

# %%
# Define our compute environment.
# This is the env that will be loaded into the docker container that does our work.
cd = CondaDependencies.create(
    pip_packages=["pandas", "numpy",
                  "azureml-sdk[explain,automl]==1.11.0", "azureml-defaults", "azureml-train-automl-runtime"],
    conda_packages=["xlrd", "scikit-learn", "numpy", "pyyaml", "pip"])
amlcompute_run_config = RunConfiguration(conda_dependencies=cd)
amlcompute_run_config.environment.docker.enabled = True

# %%
# Define Datasets
# Just noting the reference to the data store location.
iris_raw = PipelineData('iris_raw',
                        datastore=f.ws.datastores['azureml_globaldatasets'])
iris_gold = PipelineData('iris_gold',
                         datastore=f.ws.datastores['azureml_globaldatasets'])

# %%
get_iris_step = PythonScriptStep(
    name='get_iris_step',
    script_name='get_iris.py',
    arguments=['--output_dir', iris_raw],
    compute_target=f.compute_target,
    outputs=[iris_raw],
    runconfig=f.amlcompute_run_config,
    source_directory=os.path.join(
        os.getcwd(), 'pipes/get_iris'),
    allow_reuse=False
)

munge_iris_step = PythonScriptStep(
    name='munge_iris_step',
    script_name='munge_iris.py',
    arguments=['--output_dir', iris_raw],
    compute_target=f.compute_target,
    inputs=[iris_raw]
    outputs=[iris_gold],
    runconfig=f.amlcompute_run_config,
    source_directory=os.path.join(
        os.getcwd(), 'pipes/munge'),
    allow_reuse=False
)


# %% Define the pipeline
# The pipeline is a list of steps.
# The inputs and outputs of each step show where they would sit in the DAG.
pipeline = Pipeline(
    workspace=f.ws,
    steps=[get_iris_step, munge_iris_step]
)
