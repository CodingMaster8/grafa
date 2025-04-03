"""Settings for the Grafa library."""
import os
from aws_connection import get_secret_dict

#####################
# Environemnt Config
#####################

APPLICATION_ENVIRONMENT = os.getenv("APPLICATION_ENVIRONMENT", "production").lower()

application_mappings = {
    "development": "dev",
    "testing": "test",
    "staging": "stage",
    "production": "prod",
}

APPLICATION_ENVIRONMENT_SHORT = application_mappings.get(
    APPLICATION_ENVIRONMENT, APPLICATION_ENVIRONMENT
)

#LANGFUSE KEYS
langfuse_secrets = get_secret_dict("dev/grafa/langfuse")
os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_secrets["LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_SECRET_KEY"] = langfuse_secrets["LANGFUSE_SECRET_KEY"]
os.environ["LANGFUSE_HOST"] = langfuse_secrets["LANGFUSE_HOST"]

sec = get_secret_dict(f"{APPLICATION_ENVIRONMENT_SHORT}/expertai/grafa_db")

#NEO4J KEYS
grafa_uri = sec["GRAFA_URI"]
grafa_username = sec["GRAFA_USERNAME"]
grafa_password = sec["GRAFA_PASSWORD"]

os.environ["GRAFA_S3_BUCKET"] = "expertai-grafa"
