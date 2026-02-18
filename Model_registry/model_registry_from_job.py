import argparse
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

def main():
    p = argparse.ArgumentParser(description="Register model to Azure ML Model Registry from AML run")
    p.add_argument("--subscription_id", type=str, required=True, help="Azure Subscription ID")
    p.add_argument("--resource_group", type=str, required=True, help="Azure Resource Group")
    p.add_argument("--workspace_name", type=str, required=True, help="Azure ML Workspace Name")
    p.add_argument("--run-id", type=str, required=True, help="AML Run ID to register model from")
    p.add_argument("--artifact-path", type=str, required=True, help="Path to the model artifact in the AML run")
    p.add_argument("--register-model-name",type=str, required=True, help="Name to register the model under in Azure ML Model Registry")
    args = p.parse_args()

    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name
    )

    # Create model path from job run
    model_path = f"azureml://jobs/{args.run_id}/outputs/artifacts/paths/{args.artifact_path}/"
    print(f"Registering model from path: {model_path}")
    print(f"Model name: {args.register_model_name}")

    # Register the model using Azure ML SDK
    model = Model(
        path=model_path,
        name=args.register_model_name,
        description=f"Model registered from job {args.run_id}",
        type=AssetTypes.MLFLOW_MODEL
    )
    
    registered_model = ml_client.models.create_or_update(model)
    print(f"Model registered successfully!")
    print(f"  Name: {registered_model.name}")
    print(f"  Version: {registered_model.version}")
    print(f"  ID: {registered_model.id}")

if __name__ == "__main__":
    main()