from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64
import os

"""USE THIS FILE TO ENCRYPT LOGS AND DATA FILES, TO PROTECT SENSITIVE INFORMATION IN CASE OF UNAUTHORIZED ACCESS. TRY IMPLEMENTING RSA ENCRYPTION FOR ADDED SECURITY.
   USE ONCE TO GENERATE A KEY PAIR, THEN STORE THE PUBLIC AND PRIVATE KEYS IN ENVIRONMENT VARIABLES OR A SECURE VAULT.
"""

def generate_key_pair(key_size: int = 2048) -> tuple[RSA.RsaKey, RSA.RsaKey]:
    """
    Generate an RSA public/private key pair.
    """
    key = RSA.generate(key_size)
    private_key = key
    public_key = key.publickey()
    return public_key, private_key


def encrypt_message(message: str, public_key: RSA.RsaKey) -> str:
    """
    Encrypt a string message using the public key.
    Returns a Base64 encoded ciphertext.
    """
    cipher = PKCS1_OAEP.new(public_key)
    encrypted_bytes = cipher.encrypt(message.encode("utf-8"))
    return base64.b64encode(encrypted_bytes).decode("utf-8")


def decrypt_message(encrypted_base64: str, private_key: RSA.RsaKey) -> str:
    """
    Decrypt a Base64 encoded ciphertext using the private key.
    """
    cipher = PKCS1_OAEP.new(private_key)
    encrypted_bytes = base64.b64decode(encrypted_base64)
    decrypted_bytes = cipher.decrypt(encrypted_bytes)
    return decrypted_bytes.decode("utf-8")

def test(public_key, private_key) -> None:
    message = "Hello, secure world! This is a test message."
    print("Original Message:")
    print(message, "\n")

    encrypted_message = encrypt_message(message, public_key)
    print("Encrypted (Base64):")
    print(encrypted_message, "\n")

    decrypted_message = decrypt_message(encrypted_message, private_key)
    print("Decrypted Message:")
    print(decrypted_message, "\n")
    assert message == decrypted_message
    print("Encryption and decryption successful!")

def main():
    public_key, private_key = generate_key_pair()
    print("Key pair generated.\n")
    os.environ['PUBLIC_KEY'] = public_key.export_key().decode("utf-8")
    os.environ['PRIVATE_KEY'] = private_key.export_key().decode("utf-8")
    
if __name__ == "__main__":
    main()
    


    