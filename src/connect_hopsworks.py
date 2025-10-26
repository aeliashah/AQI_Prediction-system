import hopsworks
from dotenv import load_dotenv
import os

load_dotenv()

# Load secrets
project_name = os.getenv("aeliaaqipredictor")
api_key = os.getenv("uXi0plpi7JRwLdd3.3n2hQz3x9XkyQuRT6SYKp2SQ3enRvFAfeWKt6hWUfaWvXsgptpnguWomeBWG57oZ")

# Connect to Hopsworks
project = hopsworks.login(project=project_name, api_key_value=api_key)
fs = project.get_feature_store()

print("✅ Connected to Hopsworks Feature Store:", project.name)
