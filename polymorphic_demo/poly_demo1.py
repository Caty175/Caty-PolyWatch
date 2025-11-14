"""
Polymorphic Demo 1: Code Mutation Simulation
This demonstrates how polymorphic code changes its structure while maintaining functionality.
EDUCATIONAL PURPOSE ONLY - No malicious behavior
"""
import random
import hashlib
import sys

class PolymorphicDemo1:
    """Simulates polymorphic behavior through code mutation"""
    
    def __init__(self):
        self.mutation_id = random.randint(1000, 9999)
        self.junk_data = self._generate_junk()
        
    def _generate_junk(self):
        """Generate random junk data to change file signature"""
        junk_size = random.randint(100, 500)
        return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=junk_size))
    
    def calculate_hash(self):
        """Calculate hash of current instance"""
        data = f"{self.mutation_id}{self.junk_data}".encode()
        return hashlib.sha256(data).hexdigest()
    
    def execute_payload(self):
        """Benign payload - just prints information"""
        print(f"[DEMO 1] Polymorphic Demo Executing")
        print(f"[DEMO 1] Mutation ID: {self.mutation_id}")
        print(f"[DEMO 1] Hash: {self.calculate_hash()}")
        print(f"[DEMO 1] Junk Data Length: {len(self.junk_data)}")
        
        # Simulate polymorphic behavior
        operations = [
            lambda: print("[DEMO 1] Operation A: Benign calculation"),
            lambda: print("[DEMO 1] Operation B: Benign file check"),
            lambda: print("[DEMO 1] Operation C: Benign system info"),
        ]
        
        # Execute in random order
        random.shuffle(operations)
        for op in operations:
            op()
            
    def mutate(self):
        """Create a mutated version of this code"""
        print(f"[DEMO 1] Creating mutation...")
        self.mutation_id = random.randint(1000, 9999)
        self.junk_data = self._generate_junk()
        print(f"[DEMO 1] New Hash: {self.calculate_hash()}")

if __name__ == "__main__":
    print("="*60)
    print("POLYMORPHIC DEMONSTRATION 1 - CODE MUTATION")
    print("Educational Purpose Only - No Malicious Behavior")
    print("="*60)
    
    demo = PolymorphicDemo1()
    demo.execute_payload()
    
    print("\n--- Demonstrating Mutation ---")
    demo.mutate()
    demo.execute_payload()
    
    print("\n[DEMO 1] Demonstration Complete")

