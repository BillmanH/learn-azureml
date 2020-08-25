from azureml.core.compute import DataFactoryCompute
from azureml.exceptions import ComputeTargetException


def get_or_create_data_factory(workspace, factory_name):
    try:
    return DataFactoryCompute(workspace, factory_name)
    except ComputeTargetException as e:
    if 'ComputeTargetNotFound' in e.message:
    print('Data factory not found, creating...')
    provisioning_config = DataFactoryCompute.provisioning_configuration()
    data_factory = ComputeTarget.create(
        workspace, factory_name, provisioning_config)
    return data_factory
    else:
    raise e


# %%
# This is the transfer reference.
output_spot = DataReference(data_reference_name='output_data', datastore=datastore,
                            path_on_datastore='test_outputs' + '/latest', overwrite=True)

transfer_gold_step = DataTransferStep(
    name="project_outputs_gold",
    source_data_reference=shap_tables,
    destination_data_reference=output_spot,
    source_reference_type='directory',
    destination_reference_type='directory',
    compute_target=f.compute_target,
    allow_reuse=True
)
