import argparse
import mlflow
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

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

    model_uri = f"runs:/{args.run_id}/{args.artifact_path}"
    print(f"Registering model from URI: {model_uri} to Azure ML Model Registry with name: {args.register_model_name}")

    result = mlflow.register_model(model_uri,name=args.register_model_name)
    print(f"Model registered with name: {result.name} and version: {result.version}")

if __name__ == "__main__":
    main()