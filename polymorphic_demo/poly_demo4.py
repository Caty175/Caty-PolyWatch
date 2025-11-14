"""
Polymorphic Demo 4: Self-Modifying Code Simulation
This demonstrates how polymorphic code can modify itself.
EDUCATIONAL PURPOSE ONLY - No malicious behavior
"""
import random
import hashlib
import os
import sys

class PolymorphicDemo4:
    """Simulates polymorphic behavior through self-modification"""
    
    def __init__(self):
        self.generation = 1
        self.dna = self._generate_dna()
        
    def _generate_dna(self):
        """Generate unique DNA for this instance"""
        return hashlib.md5(str(random.random()).encode()).hexdigest()
    
    def calculate_signature(self):
        """Calculate current signature"""
        data = f"{self.generation}{self.dna}".encode()
        return hashlib.sha256(data).hexdigest()
    
    def execute_payload(self):
        """Benign payload execution"""
        print(f"[DEMO 4] Polymorphic Demo - Self-Modification")
        print(f"[DEMO 4] Generation: {self.generation}")
        print(f"[DEMO 4] DNA: {self.dna}")
        print(f"[DEMO 4] Signature: {self.calculate_signature()[:16]}...")
        
        # Benign operations
        self._benign_operation_1()
        self._benign_operation_2()
        self._benign_operation_3()
    
    def _benign_operation_1(self):
        """Benign operation 1"""
        result = sum(range(100))
        print(f"[DEMO 4] Operation 1: Sum calculation = {result}")
    
    def _benign_operation_2(self):
        """Benign operation 2"""
        text = "Polymorphic Demo"
        print(f"[DEMO 4] Operation 2: Text length = {len(text)}")
    
    def _benign_operation_3(self):
        """Benign operation 3"""
        print(f"[DEMO 4] Operation 3: Python version = {sys.version.split()[0]}")
    
    def mutate_self(self):
        """Simulate self-modification"""
        print(f"[DEMO 4] Mutating to next generation...")
        old_signature = self.calculate_signature()[:16]
        
        self.generation += 1
        self.dna = self._generate_dna()
        
        new_signature = self.calculate_signature()[:16]
        
        print(f"[DEMO 4] Old signature: {old_signature}...")
        print(f"[DEMO 4] New signature: {new_signature}...")
        print(f"[DEMO 4] Signatures are different: {old_signature != new_signature}")
    
    def create_offspring(self):
        """Create a new variant"""
        print(f"[DEMO 4] Creating offspring variant...")
        offspring = PolymorphicDemo4()
        offspring.generation = self.generation + 1
        print(f"[DEMO 4] Offspring created with generation {offspring.generation}")
        return offspring

if __name__ == "__main__":
    print("="*60)
    print("POLYMORPHIC DEMONSTRATION 4 - SELF-MODIFICATION")
    print("Educational Purpose Only - No Malicious Behavior")
    print("="*60)
    
    demo = PolymorphicDemo4()
    demo.execute_payload()
    
    print("\n--- Demonstrating Mutation ---")
    demo.mutate_self()
    demo.execute_payload()
    
    print("\n--- Creating Offspring ---")
    offspring = demo.create_offspring()
    offspring.execute_payload()
    
    print("\n[DEMO 4] Demonstration Complete")

