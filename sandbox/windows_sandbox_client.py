#!/usr/bin/env python3
"""
Windows Sandbox Client
Sends PE files to Windows VM, receives analysis reports

Usage:
    python windows_sandbox_client.py --vm-ip 192.168.1.100 --file malware.exe
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime


class WindowsSandboxClient:
    """Client for Windows Sandbox API Server"""
    
    def __init__(self, vm_ip, vm_port=5000):
        """
        Initialize client
        
        Args:
            vm_ip: IP address of Windows VM
            vm_port: Port of sandbox server (default: 5000)
        """
        self.vm_ip = vm_ip
        self.vm_port = vm_port
        self.base_url = f"http://{vm_ip}:{vm_port}"
    
    def health_check(self):
        """Check if server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server is healthy")
                print(f"   Service: {data.get('service')}")
                print(f"   OS: {data.get('os')}")
                print(f"   Timestamp: {data.get('timestamp')}")
                return True
            else:
                print(f"❌ Server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to {self.base_url}")
            print(f"   Make sure:")
            print(f"   1. Windows VM is running")
            print(f"   2. Sandbox server is started: python windows_sandbox_server.py")
            print(f"   3. Firewall allows port {self.vm_port}")
            print(f"   4. VM IP is correct: {self.vm_ip}")
            return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def analyze_file(self, file_path, duration=120, save_report=True):
        """
        Send file to VM for analysis
        
        Args:
            file_path: Path to PE file
            duration: Analysis duration in seconds
            save_report: Save report to local file
            
        Returns:
            dict: Analysis report
        """
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_path = Path(file_path)
            
            print(f"\n{'='*60}")
            print(f"📤 Sending file to Windows VM for analysis")
            print(f"{'='*60}")
            print(f"File: {file_path.name}")
            print(f"Size: {file_path.stat().st_size:,} bytes")
            print(f"VM: {self.vm_ip}:{self.vm_port}")
            print(f"Duration: {duration} seconds")
            print(f"{'='*60}\n")
            
            # Upload file
            print(f"📤 Uploading file...")
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'application/octet-stream')}
                data = {'duration': duration}
                
                response = requests.post(
                    f"{self.base_url}/analyze",
                    files=files,
                    data=data,
                    timeout=duration + 60  # Add buffer time
                )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Analysis complete!")
                print(f"\n{'='*60}")
                print(f"📊 ANALYSIS RESULTS")
                print(f"{'='*60}")
                print(f"Analysis ID: {result['analysis_id']}")
                print(f"Status: {result['status']}")
                print(f"Timestamp: {result['timestamp']}")
                
                # Print summary
                if 'report' in result and 'summary' in result['report']:
                    summary = result['report']['summary']
                    print(f"\n📈 Summary:")
                    print(f"   API Calls: {summary.get('api_calls', 0)}")
                    print(f"   Files Created: {summary.get('files_created', 0)}")
                    print(f"   Files Deleted: {summary.get('files_deleted', 0)}")
                    print(f"   Libraries Loaded: {summary.get('libraries_loaded', 0)}")
                    print(f"   Network Connections: {summary.get('network_connections', 0)}")
                    print(f"   DNS Queries: {summary.get('dns_queries', 0)}")
                    print(f"   Child Processes: {summary.get('child_processes', 0)}")
                
                print(f"{'='*60}\n")
                
                # Save report locally
                if save_report:
                    report_filename = f"{result['analysis_id']}_report.json"
                    with open(report_filename, 'w') as f:
                        json.dump(result['report'], f, indent=2)
                    print(f"💾 Report saved: {report_filename}")
                
                return result
            
            else:
                error = response.json()
                print(f"❌ Analysis failed!")
                print(f"   Error: {error.get('error')}")
                print(f"   Message: {error.get('message')}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"❌ Request timeout! Analysis may still be running on VM.")
            print(f"   Try retrieving the report later with --list-reports")
            return None
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_report(self, analysis_id, download=True):
        """
        Get analysis report by ID
        
        Args:
            analysis_id: Analysis ID
            download: Download report file
            
        Returns:
            dict: Analysis report
        """
        try:
            url = f"{self.base_url}/report/{analysis_id}"
            if download:
                url += "?download=true"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                if download:
                    # Save file
                    filename = f"{analysis_id}_report.json"
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Report downloaded: {filename}")
                    
                    # Load and return
                    with open(filename, 'r') as f:
                        return json.load(f)
                else:
                    return response.json()
            else:
                error = response.json()
                print(f"❌ Failed to get report: {error.get('error')}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def list_reports(self):
        """List all available reports on VM"""
        try:
            response = requests.get(f"{self.base_url}/reports", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n{'='*60}")
                print(f"📋 Available Reports ({data['count']})")
                print(f"{'='*60}\n")
                
                for report in data['reports']:
                    print(f"Analysis ID: {report['analysis_id']}")
                    print(f"  File: {report['filename']}")
                    print(f"  Timestamp: {report['timestamp']}")
                    print(f"  API Calls: {report['summary'].get('api_calls', 0)}")
                    print(f"  Files Created: {report['summary'].get('files_created', 0)}")
                    print(f"  Network Connections: {report['summary'].get('network_connections', 0)}")
                    print()
                
                return data['reports']
            else:
                print(f"❌ Failed to list reports")
                return []
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return []


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Windows Sandbox Client - Send files to Windows VM for analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if VM server is running
  python windows_sandbox_client.py --vm-ip 192.168.1.100 --health
  
  # Analyze a file
  python windows_sandbox_client.py --vm-ip 192.168.1.100 --file malware.exe
  
  # Analyze with custom duration
  python windows_sandbox_client.py --vm-ip 192.168.1.100 --file malware.exe --duration 180
  
  # List all reports on VM
  python windows_sandbox_client.py --vm-ip 192.168.1.100 --list-reports
  
  # Download specific report
  python windows_sandbox_client.py --vm-ip 192.168.1.100 --get-report <analysis_id>
        """
    )
    
    parser.add_argument(
        '--vm-ip',
        required=True,
        help='IP address of Windows VM'
    )
    
    parser.add_argument(
        '--vm-port',
        type=int,
        default=5000,
        help='Port of sandbox server (default: 5000)'
    )
    
    parser.add_argument(
        '--file',
        help='PE file to analyze'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=120,
        help='Analysis duration in seconds (default: 120)'
    )
    
    parser.add_argument(
        '--health',
        action='store_true',
        help='Check server health'
    )
    
    parser.add_argument(
        '--list-reports',
        action='store_true',
        help='List all reports on VM'
    )
    
    parser.add_argument(
        '--get-report',
        help='Download report by analysis ID'
    )
    
    args = parser.parse_args()
    
    # Create client
    client = WindowsSandboxClient(args.vm_ip, args.vm_port)
    
    # Health check
    if args.health:
        client.health_check()
        return
    
    # List reports
    if args.list_reports:
        client.list_reports()
        return
    
    # Get specific report
    if args.get_report:
        client.get_report(args.get_report, download=True)
        return
    
    # Analyze file
    if args.file:
        result = client.analyze_file(args.file, args.duration)
        
        if result:
            print(f"\n✅ Success! Next steps:")
            print(f"   1. Convert to LSTM format:")
            print(f"      python parse_behavioral_logs.py --input {result['analysis_id']}_report.json --output features.csv")
            print(f"   2. Run LSTM prediction:")
            print(f"      python ../Model/predict_lstm_behavioral.py --input features.csv")
        return
    
    # No action specified
    print(f"❌ No action specified!")
    print(f"\nUsage:")
    print(f"  python windows_sandbox_client.py --vm-ip <VM_IP> --file <file_path>")
    print(f"\nFor help:")
    print(f"  python windows_sandbox_client.py --help")


if __name__ == '__main__':
    main()

