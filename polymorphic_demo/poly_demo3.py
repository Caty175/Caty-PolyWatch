"""
Polymorphic Demo 3: Code Obfuscation Simulation
This demonstrates how polymorphic code uses obfuscation techniques.
EDUCATIONAL PURPOSE ONLY - No malicious behavior
"""
import random
import string
import time

class PolymorphicDemo3:
    """Simulates polymorphic behavior through obfuscation"""
    
    def __init__(self):
        self.obfuscation_level = random.randint(1, 5)
        self.variable_names = self._generate_random_names()
        
    def _generate_random_names(self):
        """Generate random variable names for obfuscation"""
        names = []
        for _ in range(10):
            name = ''.join(random.choices(string.ascii_letters, k=random.randint(5, 15)))
            names.append(name)
        return names
    
    def add_junk_code(self):
        """Simulate junk code insertion"""
        print(f"[DEMO 3] Inserting junk code...")
        for i in range(self.obfuscation_level):
            # Junk operations that do nothing meaningful
            _ = random.randint(1, 1000) * random.randint(1, 1000)
            _ = ''.join(random.choices(string.ascii_letters, k=50))
            time.sleep(0.01)  # Small delay
        print(f"[DEMO 3] Junk code level: {self.obfuscation_level}")
    
    def execute_with_indirection(self):
        """Execute payload with indirection"""
        # Use random variable names
        var1 = self.variable_names[0]
        var2 = self.variable_names[1]
        
        print(f"[DEMO 3] Using obfuscated variables: {var1}, {var2}")
        
        # Indirect execution path
        execution_paths = [
            self._path_a,
            self._path_b,
            self._path_c
        ]
        
        # Choose random path
        chosen_path = random.choice(execution_paths)
        chosen_path()
    
    def _path_a(self):
        """Execution path A"""
        print("[DEMO 3] Executing via Path A")
        self._common_payload()
    
    def _path_b(self):
        """Execution path B"""
        print("[DEMO 3] Executing via Path B")
        self._common_payload()
    
    def _path_c(self):
        """Execution path C"""
        print("[DEMO 3] Executing via Path C")
        self._common_payload()
    
    def _common_payload(self):
        """The actual benign payload"""
        print("[DEMO 3] Benign payload: Displaying system time")
        print(f"[DEMO 3] Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def execute_payload(self):
        """Main execution"""
        print(f"[DEMO 3] Polymorphic Demo with Obfuscation")
        self.add_junk_code()
        self.execute_with_indirection()
        print(f"[DEMO 3] Variable name set: {len(self.variable_names)} random names")

if __name__ == "__main__":
    print("="*60)
    print("POLYMORPHIC DEMONSTRATION 3 - OBFUSCATION")
    print("Educational Purpose Only - No Malicious Behavior")
    print("="*60)
    
    demo = PolymorphicDemo3()
    demo.execute_payload()
    
    print("\n--- Creating New Instance with Different Obfuscation ---")
    demo2 = PolymorphicDemo3()
    demo2.execute_payload()
    
    print("\n[DEMO 3] Demonstration Complete")

