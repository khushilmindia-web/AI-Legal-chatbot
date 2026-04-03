from __future__ import annotations

import base64
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class ApiSigningError(RuntimeError):
    pass


def generate_unique_message(url: str) -> bytes:
    timestamp_ms = int(time.time() * 1000)
    nonce = secrets.token_urlsafe(12)
    return f"{timestamp_ms}:{nonce}:{url}".encode("utf-8")


def _load_private_key(path: str | os.PathLike[str], passphrase: str | None = None) -> rsa.RSAPrivateKey:
    key_path = Path(path)
    pem_bytes = key_path.read_bytes()
    password = passphrase.encode("utf-8") if passphrase else None
    return serialization.load_pem_private_key(pem_bytes, password=password)


def _load_public_key(path: str | os.PathLike[str]) -> rsa.RSAPublicKey:
    key_path = Path(path)
    pem_bytes = key_path.read_bytes()
    return serialization.load_pem_public_key(pem_bytes)


@dataclass(frozen=True)
class SignedRequestHeaders:
    customer_id: str
    encoded_message: str
    encoded_signature: str

    def as_headers(self) -> dict[str, str]:
        return {
            "X-Customer": self.customer_id,
            "X-Message": self.encoded_message,
            "Authorization": f"HMAC {self.encoded_signature}",
        }


class ApiRequestSigner:
    def __init__(self, customer_id: str, private_key_path: str | os.PathLike[str], passphrase: str | None = None) -> None:
        self.customer_id = customer_id.strip()
        if not self.customer_id:
            raise ApiSigningError("customer_id is required")
        self.private_key_path = Path(private_key_path)
        self.passphrase = passphrase

    def sign_message(self, message: bytes) -> SignedRequestHeaders:
        private_key = _load_private_key(self.private_key_path, self.passphrase)
        signature = private_key.sign(message, padding.PKCS1v15(), hashes.SHA256())
        encoded_message = base64.b64encode(message).decode("ascii")
        encoded_signature = base64.b64encode(signature).decode("ascii")
        return SignedRequestHeaders(
            customer_id=self.customer_id,
            encoded_message=encoded_message,
            encoded_signature=encoded_signature,
        )

    def sign_request(self, request_url: str) -> SignedRequestHeaders:
        return self.sign_message(generate_unique_message(request_url))


class ApiRequestVerifier:
    def __init__(self, public_key_path: str | os.PathLike[str]) -> None:
        self.public_key_path = Path(public_key_path)

    def verify_headers(self, encoded_message: str, encoded_signature: str) -> bool:
        public_key = _load_public_key(self.public_key_path)
        message = base64.b64decode(encoded_message)
        signature = base64.b64decode(encoded_signature)
        try:
            public_key.verify(signature, message, padding.PKCS1v15(), hashes.SHA256())
            return True
        except InvalidSignature:
            return False
