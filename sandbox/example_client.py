#!/usr/bin/env python3
"""
Example Client for Sandbox API
Demonstrates how to submit files and retrieve results
"""

import requests
import time
import sys
import argparse
from pathlib import Path


class SandboxClient:
    """Client for interacting with Sandbox API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
    
    def submit_file(self, file_path, duration=120):
        """
        Submit file for analysis
        
        Args:
            file_path: Path to file to analyze
            duration: Analysis duration in seconds
            
        Returns:
            dict with analysis_id and status
        """
        url = f"{self.base_url}/analyze"
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'duration': duration}
            
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            
            return response.json()
    
    def get_status(self, analysis_id):
        """Get analysis status"""
        url = f"{self.base_url}/status/{analysis_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_result(self, analysis_id):
        """Get analysis result"""
        url = f"{self.base_url}/result/{analysis_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def download_report(self, analysis_id, output_path):
        """Download full JSON report"""
        url = f"{self.base_url}/download/{analysis_id}"
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
    
    def wait_for_completion(self, analysis_id, poll_interval=5, timeout=300):
        """
        Wait for analysis to complete
        
        Args:
            analysis_id: Analysis ID
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds
            
        Returns:
            Final status dict
        """
        start_time = time.time()
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Analysis timed out after {timeout} seconds")
            
            # Get status
            status = self.get_status(analysis_id)
            
            # Check if completed
            if status['status'] in ['completed', 'failed']:
                return status
            
            # Wait before next check
            print(f"Status: {status['status']}... waiting {poll_interval}s")
            time.sleep(poll_interval)
    
    def analyze_and_wait(self, file_path, duration=120, poll_interval=5):
        """
        Submit file and wait for results
        
        Args:
            file_path: Path to file to analyze
            duration: Analysis duration in seconds
            poll_interval: Seconds between status checks
            
        Returns:
            Analysis result dict
        """
        print(f"📤 Submitting file: {file_path}")
        
        # Submit file
        submit_result = self.submit_file(file_path, duration)
        analysis_id = submit_result['analysis_id']
        
        print(f"✅ Analysis started: {analysis_id}")
        print(f"⏱️  Estimated completion: {duration} seconds")
        print()
        
        # Wait for completion
        print("⏳ Waiting for analysis to complete...")
        status = self.wait_for_completion(analysis_id, poll_interval)
        
        if status['status'] == 'failed':
            print(f"❌ Analysis failed: {status.get('error', 'Unknown error')}")
            return None
        
        print("✅ Analysis completed!")
        print()
        
        # Get results
        print("📊 Fetching results...")
        result = self.get_result(analysis_id)
        
        return result


def print_summary(result):
    """Print analysis summary"""
    print("="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    
    summary = result.get('summary', {})
    
    print(f"\nBehavioral Features:")
    print(f"  • API Calls: {summary.get('api_calls', 0)} unique")
    print(f"  • Files Created: {summary.get('files_created', 0)}")
    print(f"  • Files Deleted: {summary.get('files_deleted', 0)}")
    print(f"  • Libraries Loaded: {summary.get('libraries_loaded', 0)}")
    print(f"  • DNS Queries: {summary.get('dns_queries', 0)}")
    
    print("\n" + "="*60)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Sandbox API Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit file and wait for results
  python3 example_client.py malware.bin
  
  # With custom server URL
  python3 example_client.py malware.bin --url http://192.168.1.100:5000
  
  # With custom duration
  python3 example_client.py malware.bin --duration 180
  
  # Download full report
  python3 example_client.py malware.bin --download report.json
        """
    )
    
    parser.add_argument(
        'file_path',
        help='Path to file to analyze'
    )
    
    parser.add_argument(
        '--url',
        default='http://localhost:5000',
        help='Sandbox API URL (default: http://localhost:5000)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=120,
        help='Analysis duration in seconds (default: 120)'
    )
    
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=5,
        help='Status check interval in seconds (default: 5)'
    )
    
    parser.add_argument(
        '--download',
        help='Download full report to this path'
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    if not Path(args.file_path).exists():
        print(f"❌ Error: File not found: {args.file_path}")
        sys.exit(1)
    
    # Create client
    client = SandboxClient(args.url)
    
    try:
        # Submit and wait for results
        result = client.analyze_and_wait(
            args.file_path,
            duration=args.duration,
            poll_interval=args.poll_interval
        )
        
        if result is None:
            sys.exit(1)
        
        # Print summary
        print_summary(result)
        
        # Download full report if requested
        if args.download:
            analysis_id = result['analysis_id']
            client.download_report(analysis_id, args.download)
            print(f"\n📄 Full report downloaded: {args.download}")
        
        print("\n✅ Analysis complete!")
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to {args.url}")
        print("   Make sure the API server is running:")
        print("   python3 sandbox_api.py --port 5000")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

