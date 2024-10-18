import os

class Config:
    def __init__(self):
        self.hostname = os.getenv('ILLUMIO_HOSTNAME', 'default_hostname')
        self.port = int(os.getenv('ILLUMIO_PORT', 8443))
        self.org_id = os.getenv('ILLUMIO_ORG_ID', 'default_org_id')
        self.api_key_id = os.getenv('ILLUMIO_API_KEY_ID', 'default_api_key_id')
        self.api_key_secret = os.getenv('ILLUMIO_API_KEY_SECRET', 'default_api_key_secret')
        self.ignore_tls = os.getenv('ILLUMIO_IGNORE_TLS', 'True').lower() in ('true', '1', 't')

config = Config()