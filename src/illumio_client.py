from illumio import *

class IllumioClient:
    def __init__(self, hostname, port, org_id, api_key_id, api_key_secret, ignore_tls=True):
        self.hostname = hostname
        self.port = port
        self.org_id = org_id
        self.api_key_id = api_key_id
        self.api_key_secret = api_key_secret
        self.ignore_tls = ignore_tls
        self.pce = PolicyComputeEngine(hostname, port=self.port, org_id=self.org_id)
        self.pce.set_credentials(self.api_key_id, self.api_key_secret)

    def get_pce(self):
        return self.pce