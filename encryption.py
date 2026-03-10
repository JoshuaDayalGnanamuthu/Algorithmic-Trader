from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import base64


def generate_key_pair(key_size: int = 2048):
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


# --- Example Usage ---
if __name__ == "__main__":

    # 1. Generate RSA keys
    public_key, private_key = generate_key_pair()
    print("Key pair generated.\n")

    # 2. Original message
    message = "Hello, secure world! This is a test message."
    print("Original Message:")
    print(message, "\n")

    # 3. Encrypt message
    encrypted_message = encrypt_message(message, public_key)
    print("Encrypted (Base64):")
    print(encrypted_message, "\n")

    # 4. Decrypt message
    decrypted_message = decrypt_message(encrypted_message, private_key)
    print("Decrypted Message:")
    print(decrypted_message, "\n")

    # 5. Verification
    assert message == decrypted_message
    print("Encryption and decryption successful!")