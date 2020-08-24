# %%
# Load Libraries

from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.core import Dataset

from azureml.pipeline.core import TrainingOutput
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.pipeline.steps import PythonScriptStep

from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import AutoMLStep
from azureml.data import OutputFileDatasetConfig

from azureml.train.automl import AutoMLConfig

# my custom libraries
import config as f

# %%
# Define our compute environment.
# This is the env that will be loaded into the docker container that does our work.
cd = CondaDependencies.create(
    pip_packages=[
        "pandas",
        "numpy",
        "azureml-sdk[explain,automl]",
        "azureml-defaults",
        "azureml-train-automl-runtime",
    ],
    conda_packages=["xlrd", "scikit-learn", "numpy", "pyyaml", "pip"],
)
amlcompute_run_config = RunConfiguration(conda_dependencies=cd)
amlcompute_run_config.environment.docker.enabled = True

# %%
# Define Datasets
# Just noting the reference to the data store location.

# saving the datastore location for consistency
datastore = f.ws.datastores[f.params["datastore_name"]]

iris_raw = PipelineData("iris_raw", datastore=datastore)


iris_gold = PipelineData("iris_gold", datastore=datastore)


# %%
# Setting up script steps

get_iris_step = PythonScriptStep(
    name="get_iris_step",
    script_name="get_iris.py",
    arguments=["--output_dir", iris_raw],
    compute_target=f.compute_target,
    outputs=[iris_raw],
    runconfig=amlcompute_run_config,
    source_directory=os.path.join(os.getcwd(), "pipes/get_iris"),
    allow_reuse=True,
)

output = OutputFileDatasetConfig(destination=(datastore, 'iris_gold'))


munge_iris_step = PythonScriptStep(
    name="munge_iris_step",
    script_name="munge_iris.py",
    arguments=["--input_dir", iris_raw, "--output_dir", output],
    compute_target=f.compute_target,
    inputs=[iris_raw],
    outputs=[output],
    runconfig=amlcompute_run_config,
    source_directory=os.path.join(os.getcwd(), "pipes/munge"),
    allow_reuse=True,
)

# register_iris_step = PythonScriptStep(
#     name="register_iris_step",
#     script_name="register_iris.py",
#     arguments=["--input_dir", iris_gold, "--register_name", "iris_gold"],
#     compute_target=f.compute_target,
#     inputs=[iris_raw]
#     runconfig=amlcompute_run_config,
#     source_directory=os.path.join(os.getcwd(), "pipes/register_iris"),
#     allow_reuse=True,
# )


# %%
# AutoML Step is very different

metrics_data = PipelineData(
    name="metrics_data",
    datastore=datastore,
    pipeline_output_name="metrics_output",
    training_output=TrainingOutput(type="Metrics"),
)

model_data = PipelineData(
    name="best_model_data",
    datastore=datastore,
    pipeline_output_name="model_output",
    training_output=TrainingOutput(type="Model"),
)

# Supported types: [azureml.data.tabular_dataset.TabularDataset,
# azureml.pipeline.core.pipeline_output_dataset.PipelineOutputTabularDataset]


automl_config = AutoMLConfig(
    training_data=output.read_delimited_files(
        path_glob="*.csv"),
    task="regression",
    experiment_timeout_minutes=60,
    label_column_name="species",
    model_explainability=False,
    compute_target=f.compute_target,
    iterations=10,
    n_cross_validations=3,
    iteration_timeout_minutes=5,
    primary_metric="spearman_correlation",
)

AutoML_step = AutoMLStep(
    "train_model",
    automl_config,
    outputs=[metrics_data, model_data],
    enable_default_model_output=False,
    enable_default_metrics_output=False,
)


# when working with outputs
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-automlstep-in-pipelines#configure-and-create-the-automated-ml-pipeline-step

# %% Define the pipeline
# The pipeline is a list of steps.
# The inputs and outputs of each step show where they would sit in the DAG.
pipeline = Pipeline(workspace=f.ws, steps=[AutoMLStep])


# %%
# Runn your model and watch the output

pipeline_run = f.exp.submit(
    pipeline, regenerate_outputs=False, continue_on_step_failure=False, tags=f.params
)

print(pipeline_run.get_portal_url())


# the output doesn't show well in Visual Studio code.
# But if running on CMD it can be useful.
# run_status = pipeline_run.wait_for_completion(show_output=True)

# %%
