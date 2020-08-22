# Model runthrough as a tutorial of the pipeline process
The infrastructure is always changing so this is here to make sure to test that the project will work before you write a bunch of code. 

by [William Jeffrey Harding](https://www.linkedin.com/in/hardingwilliam/)

william.jeffrey.harding@gmail.com 

last tested at Azure ML SDK Version:  1.11.0. It was working at that time. 

## Notebooks:
I have two notebooks that walk through the complete project of authenticating, lodading, training and looking at the output dataset. The Microsoft documentation page has similar notebooks, however mine actually work end to end. 

The notebook process assumes that you can do all of your data munging locally and in that notebook. For larger projects, with multiple transformation steps, I'd recommend the full pipleine code below. 

## Files: 

| File Name | Description |
| --- | --- |
| settings.yaml | contain all of the keys and project settings. You should be able to use AD with MFA so no need to add sensitive keys here |
| config.py | loads the settings, establishes credentials and all of the setups you will need |
| main_run.py | Defines the pipeline, and runs each individual step |


## Settings.yaml requires:
| Parameter | Description |
| --- | --- |
| workspace_name | TODO |
| expermient_name | TODO |
| compute_name | TODO |