#!/usr/bin/env python3
"""
VirtualBox Snapshot Management Utility
Handles automatic VM snapshot revert for sandbox security

Requirements:
    - VirtualBox installed
    - VBoxManage command-line tool in PATH
    - VM must have a snapshot named "Clean State" (or configured name)
"""

import os
import sys
import subprocess
import shutil
from typing import Optional, Tuple


class VirtualBoxSnapshotManager:
    """Manages VirtualBox VM snapshots for automatic revert"""
    
    def __init__(self, vm_name: str, snapshot_name: str = "Clean State"):
        """
        Initialize snapshot manager
        
        Args:
            vm_name: Name of the VirtualBox VM
            snapshot_name: Name of the snapshot to revert to (default: "Clean State")
        """
        self.vm_name = vm_name
        self.snapshot_name = snapshot_name
        self.vboxmanage_path = self._find_vboxmanage()
        
    def _find_vboxmanage(self) -> Optional[str]:
        """Find VBoxManage executable"""
        # Common installation paths
        common_paths = [
            "VBoxManage",  # In PATH
            r"C:\Program Files\Oracle\VirtualBox\VBoxManage.exe",
            r"C:\Program Files (x86)\Oracle\VirtualBox\VBoxManage.exe",
            "/usr/bin/VBoxManage",  # Linux
            "/usr/local/bin/VBoxManage",  # Linux/Mac
        ]
        
        for path in common_paths:
            if shutil.which(path) or os.path.exists(path):
                return path
        
        return None
    
    def is_available(self) -> bool:
        """Check if VBoxManage is available"""
        if not self.vboxmanage_path:
            return False
        
        try:
            result = subprocess.run(
                [self.vboxmanage_path, "--version"],
                capture_output=True,
                timeout=5,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def check_vm_exists(self) -> Tuple[bool, str]:
        """
        Check if VM exists
        
        Returns:
            (exists, message)
        """
        if not self.is_available():
            return False, "VBoxManage not found. Install VirtualBox or add to PATH."
        
        try:
            result = subprocess.run(
                [self.vboxmanage_path, "list", "vms"],
                capture_output=True,
                timeout=10,
                text=True
            )
            
            if result.returncode != 0:
                return False, f"Failed to list VMs: {result.stderr}"
            
            # Check if VM name is in the list
            vm_list = result.stdout
            if f'"{self.vm_name}"' in vm_list or f"'{self.vm_name}'" in vm_list:
                return True, "VM found"
            else:
                return False, f"VM '{self.vm_name}' not found. Available VMs:\n{vm_list}"
                
        except subprocess.TimeoutExpired:
            return False, "Timeout checking VM list"
        except Exception as e:
            return False, f"Error checking VM: {str(e)}"
    
    def check_snapshot_exists(self) -> Tuple[bool, str]:
        """
        Check if snapshot exists
        
        Returns:
            (exists, message)
        """
        if not self.is_available():
            return False, "VBoxManage not available"
        
        exists, msg = self.check_vm_exists()
        if not exists:
            return False, msg
        
        try:
            result = subprocess.run(
                [self.vboxmanage_path, "snapshot", self.vm_name, "list"],
                capture_output=True,
                timeout=10,
                text=True
            )
            
            if result.returncode != 0:
                return False, f"Failed to list snapshots: {result.stderr}"
            
            # Check if snapshot name is in the list
            snapshot_list = result.stdout
            if self.snapshot_name in snapshot_list:
                return True, "Snapshot found"
            else:
                return False, f"Snapshot '{self.snapshot_name}' not found. Available snapshots:\n{snapshot_list}"
                
        except subprocess.TimeoutExpired:
            return False, "Timeout checking snapshots"
        except Exception as e:
            return False, f"Error checking snapshot: {str(e)}"
    
    def revert_to_snapshot(self) -> Tuple[bool, str]:
        """
        Revert VM to snapshot
        
        Returns:
            (success, message)
        """
        if not self.is_available():
            return False, "VBoxManage not found. Cannot revert snapshot."
        
        # Check if VM exists
        exists, msg = self.check_vm_exists()
        if not exists:
            return False, f"Cannot revert: {msg}"
        
        # Check if snapshot exists
        exists, msg = self.check_snapshot_exists()
        if not exists:
            return False, f"Cannot revert: {msg}"
        
        try:
            print(f"🔄 Reverting VM '{self.vm_name}' to snapshot '{self.snapshot_name}'...")
            
            # First, power off VM if running
            result = subprocess.run(
                [self.vboxmanage_path, "controlvm", self.vm_name, "poweroff"],
                capture_output=True,
                timeout=30,
                text=True
            )
            # Ignore errors - VM might already be off
            
            # Wait a moment for VM to shut down
            import time
            time.sleep(2)
            
            # Revert to snapshot
            result = subprocess.run(
                [self.vboxmanage_path, "snapshot", self.vm_name, "restore", self.snapshot_name],
                capture_output=True,
                timeout=60,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully reverted VM to snapshot '{self.snapshot_name}'")
                return True, f"VM reverted to snapshot '{self.snapshot_name}'"
            else:
                error_msg = result.stderr or result.stdout
                return False, f"Failed to revert snapshot: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return False, "Timeout reverting snapshot (VM may be busy)"
        except Exception as e:
            return False, f"Error reverting snapshot: {str(e)}"
    
    def get_status(self) -> dict:
        """Get status information"""
        vbox_available = self.is_available()
        vm_exists, vm_msg = self.check_vm_exists() if vbox_available else (False, "VBoxManage not available")
        snapshot_exists, snapshot_msg = self.check_snapshot_exists() if vm_exists else (False, "VM not found")
        
        return {
            "vboxmanage_available": vbox_available,
            "vm_name": self.vm_name,
            "vm_exists": vm_exists,
            "vm_message": vm_msg,
            "snapshot_name": self.snapshot_name,
            "snapshot_exists": snapshot_exists,
            "snapshot_message": snapshot_msg,
            "ready": vbox_available and vm_exists and snapshot_exists
        }


def main():
    """CLI for testing snapshot management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VirtualBox Snapshot Manager')
    parser.add_argument('--vm-name', required=True, help='VM name')
    parser.add_argument('--snapshot-name', default='Clean State', help='Snapshot name (default: Clean State)')
    parser.add_argument('--check', action='store_true', help='Check if VM and snapshot exist')
    parser.add_argument('--revert', action='store_true', help='Revert to snapshot')
    parser.add_argument('--status', action='store_true', help='Show status')
    
    args = parser.parse_args()
    
    manager = VirtualBoxSnapshotManager(args.vm_name, args.snapshot_name)
    
    if args.status:
        status = manager.get_status()
        print(f"\n{'='*60}")
        print(f"VirtualBox Snapshot Manager Status")
        print(f"{'='*60}")
        print(f"VBoxManage Available: {status['vboxmanage_available']}")
        print(f"VM Name: {status['vm_name']}")
        print(f"VM Exists: {status['vm_exists']}")
        print(f"VM Message: {status['vm_message']}")
        print(f"Snapshot Name: {status['snapshot_name']}")
        print(f"Snapshot Exists: {status['snapshot_exists']}")
        print(f"Snapshot Message: {status['snapshot_message']}")
        print(f"Ready: {status['ready']}")
        print(f"{'='*60}\n")
    
    if args.check:
        vm_exists, vm_msg = manager.check_vm_exists()
        print(f"VM Check: {vm_msg}")
        
        if vm_exists:
            snapshot_exists, snapshot_msg = manager.check_snapshot_exists()
            print(f"Snapshot Check: {snapshot_msg}")
    
    if args.revert:
        success, msg = manager.revert_to_snapshot()
        if success:
            print(f"✅ {msg}")
        else:
            print(f"❌ {msg}")
            sys.exit(1)


if __name__ == '__main__':
    main()

