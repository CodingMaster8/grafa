import os
import json
import botocore
import botocore.session
from aws_secretsmanager_caching imoprt SecretCache, SecretCacheConfig

#####################
# AWS Connection
#####################

client = botocore.session.get_session().create_client('secretsmanager')
cache_config = SecretCacheConfig

SECRETS_CACHE = SecretCache( config = cache_config, client = client)

def get_local_secret(secret_str):
    secret = os.getenv("SECRET__"+secret_str)
    if secret is None:
        raise KeyError(f'"SECRET__{secret_str}" was not found in the Environment Variables')
    return secret

def get_secret(secret_str):
    try:
        return get_local_secret(secret_str)
    except:
        pass
    try:
        return SECRETS_CACHE.get_secret_string(secret_str)
    except Exception as e:
        return KeyError(f'Secret "{secret_str}" not available locally and Secret Manager returned an error {e}')

def get_secret_dict(secret_str):
    return json.loads(get_secret(secret_str))