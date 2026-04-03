# RSA API Key Integration

This project now includes a production-ready RSA signing utility and a CLI generator for API integrations that require public-private key authentication.

## Files

- `scripts/generate_api_keypair.py`
- `backend/app/utils/api_signing.py`

## Environment Variables

Add these values to `.env`:

```env
API_CUSTOMER_ID=your-email@example.com
API_PRIVATE_KEY_PATH=secrets/api_private_key.pem
API_PUBLIC_KEY_PATH=secrets/api_public_key.pem
```

Optional:

```env
API_PRIVATE_KEY_PASSPHRASE=change-me
```

## Generate a 2048-bit RSA key pair

```powershell
.\venv\Scripts\python.exe scripts\generate_api_keypair.py
```

This generates:

- a private key in PEM/PKCS8 format
- a public key in PEM format

## Python Example

```python
import os

from backend.app.utils.api_signing import ApiRequestSigner, ApiRequestVerifier

customer_id = os.environ["API_CUSTOMER_ID"]
private_key_path = os.environ["API_PRIVATE_KEY_PATH"]
public_key_path = os.environ["API_PUBLIC_KEY_PATH"]

request_url = "https://api.example.com/search/?q=cheque+bounce"

signer = ApiRequestSigner(
    customer_id=customer_id,
    private_key_path=private_key_path,
    passphrase=os.getenv("API_PRIVATE_KEY_PASSPHRASE"),
)
signed = signer.sign_request(request_url)
headers = signed.as_headers()

verifier = ApiRequestVerifier(public_key_path=public_key_path)
is_valid = verifier.verify_headers(
    encoded_message=signed.encoded_message,
    encoded_signature=signed.encoded_signature,
)

print(headers)
print(is_valid)
```

Example request headers:

```text
X-Customer: your-email@example.com
X-Message: <base64 encoded unique message>
Authorization: HMAC <base64 encoded signature>
```

## Security Best Practices

- Never hardcode private keys, passphrases, or customer IDs in source files.
- Store key paths in `.env`, not the key contents.
- Keep private keys outside Git and outside public/static directories.
- If possible, encrypt the private key with `API_PRIVATE_KEY_PASSPHRASE`.
- Rotate keys if they are exposed.
- Restrict private key file permissions to the service user only.
- Use separate key pairs for development and production.
- Log only key paths or customer IDs, never private key contents or signatures.
