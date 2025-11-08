from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment
from azure.ai.ml import command

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="",
    resource_group_name="object_manipulation",
    workspace_name="CV-workspace"
)

env = Environment(
    name="swinunetr-env",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8"
)

ml_client.environments.create_or_update(env)

job = command(
    code="./",
    command="python train.py --data_path /home/azureuser/cloudfiles/code/projects/BRATS_3D/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData --epochs 50",
    environment="swinunetr-env@latest",
    compute="NC4as-T4-V3-Cluster",
    display_name="brats-swinunetr",
    experiment_name="brats_swinunetr_training"
)

returned_job = ml_client.jobs.create_or_update(job)
print(returned_job.studio_url)
