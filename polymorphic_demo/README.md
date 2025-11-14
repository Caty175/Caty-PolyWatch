# Polymorphic Malware Demonstration Files

**EDUCATIONAL PURPOSE ONLY - NO MALICIOUS BEHAVIOR**

This folder contains demonstration files that simulate polymorphic malware characteristics for testing sandbox systems. All files are completely benign and only demonstrate the *concepts* of polymorphic behavior.

## Files Included

### Python PE Files (4 files)
1. **poly_demo1.py** - Code Mutation Simulation
   - Demonstrates how polymorphic code changes structure while maintaining functionality
   - Uses random mutation IDs and junk data insertion
   - Shows hash changes between mutations

2. **poly_demo2.py** - Encryption/Decryption Simulation
   - Demonstrates simple XOR encryption to change file appearance
   - Each instance uses different encryption keys
   - Shows how encrypted payloads look different but decrypt to same functionality

3. **poly_demo3.py** - Code Obfuscation Simulation
   - Demonstrates obfuscation techniques (junk code, random variable names)
   - Uses random execution paths
   - Shows code indirection methods

4. **poly_demo4.py** - Self-Modifying Code Simulation
   - Demonstrates self-mutation capabilities
   - Shows generation tracking and DNA changes
   - Creates offspring variants with different signatures

### Other File Types (2 files)
5. **poly_demo.ps1** - PowerShell Polymorphic Demo
   - Demonstrates polymorphic behavior in PowerShell
   - Includes encryption, random paths, and mutation

6. **poly_demo.bat** - Batch Script Polymorphic Demo
   - Demonstrates polymorphic behavior in batch files
   - Shows random execution paths and encoding simulation

## Polymorphic Characteristics Demonstrated

Each file demonstrates one or more of these polymorphic techniques:

- **Signature Variation**: Each execution produces different file signatures/hashes
- **Code Mutation**: Random junk data and variable names change the code structure
- **Encryption**: Simple encryption makes the payload look different each time
- **Obfuscation**: Random execution paths and junk operations obscure the code flow
- **Self-Modification**: Code can mutate itself to create new variants

## How to Convert Python Files to PE Executables

To create actual PE (.exe) files for testing in your sandbox:

### Using PyInstaller (Recommended)

```bash
# Install PyInstaller
pip install pyinstaller

# Convert each Python file to a standalone executable
pyinstaller --onefile poly_demo1.py
pyinstaller --onefile poly_demo2.py
pyinstaller --onefile poly_demo3.py
pyinstaller --onefile poly_demo4.py
```

The executables will be created in the `dist/` folder.

### Alternative: Using cx_Freeze

```bash
# Install cx_Freeze
pip install cx_Freeze

# Create executables
cxfreeze poly_demo1.py --target-dir dist/demo1
cxfreeze poly_demo2.py --target-dir dist/demo2
cxfreeze poly_demo3.py --target-dir dist/demo3
cxfreeze poly_demo4.py --target-dir dist/demo4
```

## Running the Demos

### Python Files
```bash
python poly_demo1.py
python poly_demo2.py
python poly_demo3.py
python poly_demo4.py
```

### PowerShell Script
```powershell
.\poly_demo.ps1
```

### Batch Script
```cmd
poly_demo.bat
```

## Testing in Your Sandbox

1. **Convert Python files to PE executables** using PyInstaller
2. **Copy all files** (4 PE files + 2 script files) to your sandbox environment
3. **Run each file** and observe the polymorphic behavior
4. **Monitor** how your sandbox system detects and analyzes these files
5. **Compare signatures** between multiple runs of the same file

## Expected Behavior

All files will:
- Print demonstration messages to the console
- Show their polymorphic characteristics (mutation IDs, hashes, encryption keys)
- Execute completely benign operations (calculations, system info display)
- Demonstrate that each run produces different signatures
- **NOT** perform any malicious actions

## Safety Notes

- ✅ All files are completely safe and benign
- ✅ No network connections are made
- ✅ No files are modified or created (except by PyInstaller during compilation)
- ✅ No system changes are made
- ✅ No data is collected or exfiltrated
- ✅ Suitable for educational and testing purposes

## Use Cases

- Testing sandbox detection capabilities
- Understanding polymorphic malware concepts
- Training security analysts
- Developing malware detection systems
- Educational demonstrations

---

**Remember**: These are educational demonstrations only. Real polymorphic malware is dangerous and illegal to create or distribute.

