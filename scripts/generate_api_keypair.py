from __future__ import annotations

import os
import stat
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def set_private_key_permissions(private_key_path: Path) -> None:
    if os.name == "nt":
        return
    private_key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def generate_rsa_keypair(
    private_key_path: Path,
    public_key_path: Path,
    passphrase: str | None = None,
) -> None:
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    public_key_path.parent.mkdir(parents=True, exist_ok=True)

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    encryption = (
        serialization.BestAvailableEncryption(passphrase.encode("utf-8"))
        if passphrase
        else serialization.NoEncryption()
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    private_key_path.write_bytes(private_pem)
    public_key_path.write_bytes(public_pem)
    set_private_key_permissions(private_key_path)


def main() -> None:
    private_key_path = Path(os.getenv("API_PRIVATE_KEY_PATH", "secrets/api_private_key.pem"))
    public_key_path = Path(os.getenv("API_PUBLIC_KEY_PATH", "secrets/api_public_key.pem"))
    passphrase = os.getenv("API_PRIVATE_KEY_PASSPHRASE") or None

    if private_key_path.exists() or public_key_path.exists():
        raise SystemExit(
            "Refusing to overwrite existing key files. Delete them first or change API_PRIVATE_KEY_PATH/API_PUBLIC_KEY_PATH."
        )

    generate_rsa_keypair(private_key_path=private_key_path, public_key_path=public_key_path, passphrase=passphrase)
    print(f"Private key saved to: {private_key_path}")
    print(f"Public key saved to:  {public_key_path}")
    print("Keep the private key out of source control and load paths from .env.")


if __name__ == "__main__":
    main()
