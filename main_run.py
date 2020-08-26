# %%
# Load Libraries

# my custom libraries

import config as f
from azureml.core import Dataset

from azureml.data import OutputFileDatasetConfig
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData, TrainingOutput
from azureml.pipeline.steps import (AutoMLStep, DataTransferStep,
                                    PythonScriptStep)
from azureml.train.automl import AutoMLConfig


# %%
# Define Datasets
# Just noting the reference to the data store location.

# saving the datastore location for consistency
datastore = f.ws.datastores[f.params["datastore_name"]]
iris_raw = PipelineData("iris_raw", datastore=datastore)
iris_gold = PipelineData("iris_gold", datastore=datastore)
shap_tables = PipelineData("shap_tables", datastore=datastore)


# %%
# Setting up script steps

get_iris_step = PythonScriptStep(
    name="get_iris_step",
    script_name="get_iris.py",
    arguments=["--output_dir", iris_raw],
    compute_target=f.compute_target,
    outputs=[iris_raw],
    runconfig=f.amlcompute_run_config,
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
    runconfig=f.amlcompute_run_config,
    source_directory=os.path.join(os.getcwd(), "pipes/munge"),
    allow_reuse=True,
)


# %%
# AutoML Step is set up separately.

metrics_data = PipelineData(
    name="metrics_data_json",
    datastore=datastore,
    pipeline_output_name="metrics_output",
    training_output=TrainingOutput(type="Metrics")
)

model_data = PipelineData(
    name="best_model_pkl",
    datastore=datastore,
    pipeline_output_name="model_output",
    training_output=TrainingOutput(type="Model"),
)

# Supported types: [azureml.data.tabular_dataset.TabularDataset,
# azureml.pipeline.core.pipeline_output_dataset.PipelineOutputTabularDataset]

automl_settings = {
    "iteration_timeout_minutes": 5,
    "iterations": 1,
    "n_cross_validations": 2,
    "primary_metric": 'accuracy',
    "featurization": 'auto',
    "max_concurrent_iterations": 5
}

automl_config = AutoMLConfig(task='classification',
                             debug_log='automl_errors.log',
                             path='iris_gold',
                             training_data=output.read_delimited_files(
                                 'iris_gold.csv'),
                             label_column_name="species",
                             compute_target=f.compute_target,
                             model_explainability=True,
                             ** automl_settings)

train_step = AutoMLStep('automl', automl_config,
                        outputs=[metrics_data, model_data],
                        enable_default_model_output=False,
                        enable_default_metrics_output=False,
                        allow_reuse=True,
                        passthru_automl_config=False)


# %%

score_step = PythonScriptStep(
    name="score_step",
    script_name="score_step.py",
    arguments=["--model_data", model_data, "--iris_gold", output],
    compute_target=f.compute_target,
    inputs=[model_data,  output],
    outputs=[shap_tables],
    runconfig=f.amlcompute_run_config,
    source_directory=os.path.join(os.getcwd(), "pipes/score_step"),
    allow_reuse=True,
)


# when working with outputs
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-automlstep-in-pipelines#configure-and-create-the-automated-ml-pipeline-step

# %% Define the pipeline
# The pipeline is a list of steps.
# The inputs and outputs of each step show where they would sit in the DAG.
pipeline = Pipeline(workspace=f.ws, steps=[score_step])


# %%
# Runn your model and watch the output

pipeline_run = f.exp.submit(
    pipeline, regenerate_outputs=False, continue_on_step_failure=False, tags=f.params
)

# print(pipeline_run.get_portal_url())
# pipeline_run.wait_for_completion()

# the output doesn't show well in Visual Studio code.
# But if running on CMD it can be useful.
# run_status = pipeline_run.wait_for_completion(show_output=True)

# %%
