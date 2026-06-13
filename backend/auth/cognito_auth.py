"""
QuantEdge v6.0 — Auth compatibility shim.

The platform moved off AWS Cognito to self-hosted local JWT auth. This module
used to hold the Cognito client; it now re-exports the local implementation so
every existing `from auth.cognito_auth import ...` keeps working with no router
changes. See auth/local_auth.py for the implementation and the AuthProvider
seam for future multi-user.

Original Cognito version preserved at: auth/cognito_auth.py.aws-backup
"""

from auth.local_auth import (  # noqa: F401
    CognitoUser,
    CognitoAuthenticator,
    get_current_user,
    get_optional_user,
    get_cognito_auth,
    verify_cognito_token,
)
