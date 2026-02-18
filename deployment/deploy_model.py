"""
Deploy diabetes model to Azure ML Online Endpoint
"""
import os
import sys
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    OnlineRequestSettings,
    ProbeSettings
)
from azure.identity import DefaultAzureCredential

def create_ml_client(subscription_id, resource_group, workspace_name):
    """Create Azure ML client"""
    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        print(f"✅ Connected to workspace: {ml_client.workspace_name}")
        return ml_client
    except Exception as e:
        print(f"❌ Error connecting to Azure ML: {e}")
        sys.exit(1)

def validate_model_exists(ml_client, model_reference):
    """Validate that model exists in registry"""
    try:
        # Parse model reference
        model_name = model_reference.replace('@latest', '').replace(':latest', '')
        
        print(f"🔍 Validating model: {model_name}")
        
        # Check if @latest or specific version
        if '@latest' in model_reference or ':latest' in model_reference:
            # List models to get latest version (sorted descending by version)
            models = list(ml_client.models.list(name=model_name))
            if models:
                latest = models[0]
                print(f"✅ Found model: {latest.name} v{latest.version}")
                return f"{latest.name}:{latest.version}"
        else:
            # Specific version
            if ':' in model_reference:
                name, version = model_reference.split(':')
            elif '@' in model_reference:
                name, version = model_reference.split('@')
            else:
                name, version = model_reference, '1'
            
            model = ml_client.models.get(name=name, version=version)
            print(f"✅ Found model: {model.name} v{model.version}")
            return f"{model.name}:{model.version}"
        
        print(f"⚠️ Model not found, using reference as-is: {model_reference}")
        return model_reference.replace('@', ':')
        
    except Exception as e:
        print(f"⚠️ Could not validate model: {e}")
        print(f"💡 Proceeding with reference: {model_reference}")
        return model_reference.replace('@', ':')

def validate_endpoint_exists(ml_client, endpoint_name):
    """Validate that endpoint exists"""
    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        print(f"✅ Endpoint found: {endpoint_name}")
        print(f"   State: {endpoint.provisioning_state}")
        
        if endpoint.provisioning_state != "Succeeded":
            print(f"⚠️  Warning: Endpoint state is '{endpoint.provisioning_state}'")
            return None
        
        return endpoint
    except Exception as e:
        print(f"❌ Endpoint '{endpoint_name}' not found: {e}")
        sys.exit(1)

def delete_deployment_if_exists(ml_client, endpoint_name, deployment_name):
    """Delete deployment if it exists"""
    try:
        deployment = ml_client.online_deployments.get(
            name=deployment_name,
            endpoint_name=endpoint_name
        )
        print(f"⚠️  Deployment '{deployment_name}' already exists. Deleting...")
        ml_client.online_deployments.begin_delete(
            name=deployment_name,
            endpoint_name=endpoint_name
        ).result()
        print(f"✅ Deployment deleted")
    except Exception:
        print(f"✅ Deployment '{deployment_name}' does not exist")

def create_deployment(ml_client, endpoint_name, deployment_name, model_name, 
                     instance_type, instance_count):
    """Create deployment"""
    try:
        print(f"🚀 Creating deployment: {deployment_name}")
        print(f"   Model: {model_name}")
        print(f"   Instance: {instance_type} x{instance_count}")
        
        # Configure request settings
        request_settings = OnlineRequestSettings(
            request_timeout_ms=90000,
            max_concurrent_requests_per_instance=1,
            max_queue_wait_ms=60000
        )
        
        # Configure health probes
        liveness_probe = ProbeSettings(
            failure_threshold=3,
            success_threshold=1,
            timeout=2,
            period=10,
            initial_delay=10
        )
        
        readiness_probe = ProbeSettings(
            failure_threshold=3,
            success_threshold=1,
            timeout=10,
            period=10,
            initial_delay=10
        )
        
        # Create deployment configuration
        # Using same curated environment as training to avoid MLflow auto-build failures
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=model_name,
            environment="azureml://registries/azureml/environments/AzureML-sklearn-0.24-ubuntu18.04-py37-cpu/labels/latest",
            instance_type=instance_type,
            instance_count=instance_count,
            request_settings=request_settings,
            liveness_probe=liveness_probe,
            readiness_probe=readiness_probe
        )
        
        # Deploy
        print(f"⏳ Deploying (this may take several minutes)...")
        ml_client.online_deployments.begin_create_or_update(deployment).result()
        
        print(f"✅ Deployment created successfully")
        
        # Set traffic to 100%
        print(f"🔄 Setting traffic to 100%...")
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        print(f"✅ Traffic configured")
        return True
        
    except Exception as e:
        print(f"❌ Error creating deployment: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to get logs if deployment failed
        try:
            print(f"\n📋 Attempting to retrieve logs...")
            logs = ml_client.online_deployments.get_logs(
                name=deployment_name,
                endpoint_name=endpoint_name,
                lines=100
            )
            print(f"\n🔍 DEPLOYMENT LOGS:\n{logs}")
        except:
            pass
        
        return False

def main():
    # Get environment variables
    subscription_id = os.environ.get('SUBSCRIPTION_ID')
    resource_group = os.environ.get('RESOURCE_GROUP')
    workspace_name = os.environ.get('WORKSPACE_NAME')
    endpoint_name = os.environ.get('ENDPOINT_NAME', 'diabetes-endpoint')
    deployment_name = os.environ.get('DEPLOYMENT_NAME', 'blue')
    model_name = os.environ.get('MODEL_NAME', 'diabetes-model@latest')
    instance_type = os.environ.get('INSTANCE_TYPE', 'Standard_DS3_v2')
    instance_count = int(os.environ.get('INSTANCE_COUNT', '1'))
    
    print("=" * 60)
    print("🚀 DIABETES MODEL DEPLOYMENT")
    print("=" * 60)
    print(f"Subscription: {subscription_id}")
    print(f"Resource Group: {resource_group}")
    print(f"Workspace: {workspace_name}")
    print(f"Endpoint: {endpoint_name}")
    print(f"Deployment: {deployment_name}")
    print(f"Model: {model_name}")
    print("=" * 60)
    
    # Validate required variables
    if not all([subscription_id, resource_group, workspace_name]):
        print("❌ Missing required environment variables")
        print("   Required: SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME")
        sys.exit(1)
    
    # Create ML client
    ml_client = create_ml_client(subscription_id, resource_group, workspace_name)
    
    # Validate endpoint exists
    validate_endpoint_exists(ml_client, endpoint_name)
    
    # Validate model exists and get full reference
    validated_model = validate_model_exists(ml_client, model_name)
    
    print(f"\n📦 Using model: {validated_model}")
    
    # Delete existing deployment if exists
    delete_deployment_if_exists(ml_client, endpoint_name, deployment_name)
    
    # Create new deployment
    success = create_deployment(
        ml_client, endpoint_name, deployment_name, validated_model,
        instance_type, instance_count
    )
    
    if success:
        print("\n🎉 DEPLOYMENT COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("\n❌ DEPLOYMENT FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
