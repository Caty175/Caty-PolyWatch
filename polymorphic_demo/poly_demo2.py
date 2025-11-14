"""
Polymorphic Demo 2: Encryption/Decryption Simulation
This demonstrates how polymorphic code uses encryption to change appearance.
EDUCATIONAL PURPOSE ONLY - No malicious behavior
"""
import base64
import random
import os

class PolymorphicDemo2:
    """Simulates polymorphic behavior through encryption"""
    
    def __init__(self):
        self.encryption_key = random.randint(1, 255)
        self.encrypted_payload = None
        
    def simple_xor_encrypt(self, data, key):
        """Simple XOR encryption for demonstration"""
        return bytes([b ^ key for b in data])
    
    def encrypt_payload(self, payload):
        """Encrypt the payload"""
        payload_bytes = payload.encode()
        encrypted = self.simple_xor_encrypt(payload_bytes, self.encryption_key)
        self.encrypted_payload = base64.b64encode(encrypted).decode()
        return self.encrypted_payload
    
    def decrypt_payload(self):
        """Decrypt the payload"""
        if not self.encrypted_payload:
            return None
        encrypted_bytes = base64.b64decode(self.encrypted_payload)
        decrypted = self.simple_xor_encrypt(encrypted_bytes, self.encryption_key)
        return decrypted.decode()
    
    def execute_payload(self):
        """Benign payload execution"""
        print(f"[DEMO 2] Polymorphic Demo with Encryption")
        print(f"[DEMO 2] Encryption Key: {self.encryption_key}")
        
        # Original payload (benign message)
        original = "This is a benign demonstration payload"
        print(f"[DEMO 2] Original: {original}")
        
        # Encrypt
        encrypted = self.encrypt_payload(original)
        print(f"[DEMO 2] Encrypted: {encrypted}")
        
        # Decrypt and execute
        decrypted = self.decrypt_payload()
        print(f"[DEMO 2] Decrypted: {decrypted}")
        
        # Demonstrate that each instance has different encryption
        print(f"[DEMO 2] Each instance uses different key: {self.encryption_key}")
        
    def generate_variant(self):
        """Generate a new variant with different encryption"""
        print(f"[DEMO 2] Generating new variant...")
        old_key = self.encryption_key
        self.encryption_key = random.randint(1, 255)
        print(f"[DEMO 2] Key changed from {old_key} to {self.encryption_key}")

if __name__ == "__main__":
    print("="*60)
    print("POLYMORPHIC DEMONSTRATION 2 - ENCRYPTION")
    print("Educational Purpose Only - No Malicious Behavior")
    print("="*60)
    
    demo = PolymorphicDemo2()
    demo.execute_payload()
    
    print("\n--- Generating New Variant ---")
    demo.generate_variant()
    demo.execute_payload()
    
    print("\n[DEMO 2] Demonstration Complete")

