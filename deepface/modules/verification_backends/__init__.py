# project dependencies
from deepface.modules.verification_backends.base import VerificationBackend
from deepface.modules.verification_backends.factory import create_verification_backend

__all__ = ['VerificationBackend', 'create_verification_backend']