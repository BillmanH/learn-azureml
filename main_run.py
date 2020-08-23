# %%
# Load Libraries
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.data.data_reference import DataReference

# from azureml.pipeline.steps import AutoMLStep

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
        "azureml-sdk[explain,automl]==1.11.0",
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
iris_raw = PipelineData(
    "iris_raw", datastore=f.ws.datastores[f.params["datastore_name"]]
)
iris_gold = PipelineData(
    "iris_gold", datastore=f.ws.datastores[f.params["datastore_name"]]
)

# %%
get_iris_step = PythonScriptStep(
    name="get_iris_step",
    script_name="get_iris.py",
    arguments=["--output_dir", iris_raw],
    compute_target=f.compute_target,
    outputs=[iris_raw],
    runconfig=amlcompute_run_config,
    source_directory=os.path.join(os.getcwd(), "pipes/get_iris"),
    allow_reuse=False,
)

munge_iris_step = PythonScriptStep(
    name="munge_iris_step",
    script_name="munge_iris.py",
    arguments=["--input_dir", iris_raw, "--output_dir", iris_gold],
    compute_target=f.compute_target,
    inputs=[iris_raw],
    outputs=[iris_gold],
    runconfig=amlcompute_run_config,
    source_directory=os.path.join(os.getcwd(), "pipes/munge"),
    allow_reuse=False,
)

# %%
# The AutoML model is a little different and requires a separate run configuration.

# automl_config = AutoMLConfig(
#     task="regression",
#     experiment_timeout_minutes=60,
#     training_data=training_data,
#     label_column_name=target,
#     model_explainability=True,
#     iterations=10,
#     n_cross_validations=3,
#     iteration_timeout_minutes=5,
#     primary_metric="spearman_correlation",
# )

# AutoML_step = AutoMLStep("train_model", automl_config)


# %% Define the pipeline
# The pipeline is a list of steps.
# The inputs and outputs of each step show where they would sit in the DAG.
pipeline = Pipeline(workspace=f.ws, steps=[get_iris_step, munge_iris_step])


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
