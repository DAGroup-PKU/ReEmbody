"""Freeze the configured remote model/route, without recording credentials.

A provider alias may change its backend; this records the observable API
identity and does not attest to immutable remote weights.
"""
import hashlib
import os
from common import load_credentials


def api_identity():
    model=os.environ.get('KIMI_MODEL','Kimi-K3')
    endpoint=os.environ.get('KIMI_BASE_URL','')
    if not model or not endpoint:raise ValueError('Configured K3 model and base URL are required')
    return {'model':model,'base_url_sha256':hashlib.sha256(endpoint.encode()).hexdigest(),
            'credentials_recorded':False,'provider_weights_immutability_verified':False}


def capture_api_identity(campaign):
    load_credentials(campaign['runtime']['env_file'])
    return api_identity()


def assert_api_identity(expected):
    if api_identity()!=expected:
        raise RuntimeError('Frozen K3 model or API route changed; start a new revision')
