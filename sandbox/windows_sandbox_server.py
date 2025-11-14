#!/usr/bin/env python3
"""
Windows Sandbox API Server
Receives PE files from host, runs sandbox analysis, returns logs

Usage:
    python windows_sandbox_server.py --port 5000 --host 0.0.0.0
"""

import os
import sys
import json
import uuid
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Import the sandbox
from windows_sandbox import WindowsSandbox

# Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path("uploads")
LOGS_FOLDER = Path("logs")
ALLOWED_EXTENSIONS = {'.exe', '.dll', '.sys', '.bat', '.ps1', '.vbs', '.js', '.jar', '.scr', '.com'}

# Create directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
LOGS_FOLDER.mkdir(exist_ok=True)

# Global config
config = {
    'default_duration': 120,
    'max_duration': 600,
    'auto_cleanup': True
}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Windows Sandbox Server',
        'timestamp': datetime.now().isoformat(),
        'os': sys.platform,
        'upload_folder': str(UPLOAD_FOLDER.absolute()),
        'logs_folder': str(LOGS_FOLDER.absolute())
    })


@app.route('/analyze', methods=['POST'])
def analyze_file():
    """
    Analyze uploaded file
    
    Request:
        - file: PE file to analyze (multipart/form-data)
        - duration: Analysis duration in seconds (optional, default: 120)
        
    Response:
        - analysis_id: Unique analysis ID
        - status: Analysis status
        - report: Behavioral analysis report (JSON)
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please upload a file using the "file" field'
            }), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                'error': 'Empty filename',
                'message': 'Please provide a valid filename'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': f'Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
                'received': Path(file.filename).suffix
            }), 400
        
        # Get duration parameter
        duration = request.form.get('duration', config['default_duration'])
        try:
            duration = int(duration)
            if duration > config['max_duration']:
                duration = config['max_duration']
        except ValueError:
            duration = config['default_duration']
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = UPLOAD_FOLDER / f"{analysis_id}_{filename}"
        file.save(str(upload_path))
        
        print(f"📁 File uploaded: {filename}")
        print(f"🆔 Analysis ID: {analysis_id}")
        print(f"⏱️  Duration: {duration} seconds")
        
        # Run sandbox analysis
        sandbox = WindowsSandbox(duration=duration, output_dir=str(LOGS_FOLDER))
        
        # Output file path
        report_path = LOGS_FOLDER / f"{analysis_id}_report.json"
        
        # Run analysis
        report = sandbox.run_in_sandbox(str(upload_path), str(report_path))
        
        # Cleanup uploaded file if configured
        if config['auto_cleanup']:
            try:
                os.remove(upload_path)
                print(f"🗑️  Cleaned up: {upload_path}")
            except:
                pass
        
        # Return report
        return jsonify({
            'analysis_id': analysis_id,
            'status': 'completed',
            'filename': filename,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'report': report,
            'report_file': str(report_path)
        }), 200
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/report/<analysis_id>', methods=['GET'])
def get_report(analysis_id):
    """
    Get analysis report by ID
    
    Parameters:
        - analysis_id: Analysis ID
        
    Response:
        - JSON report or file download
    """
    try:
        # Find report file
        report_files = list(LOGS_FOLDER.glob(f"{analysis_id}_report.json"))
        
        if not report_files:
            return jsonify({
                'error': 'Report not found',
                'analysis_id': analysis_id
            }), 404
        
        report_path = report_files[0]
        
        # Check if client wants download or JSON
        if request.args.get('download') == 'true':
            return send_file(
                str(report_path),
                mimetype='application/json',
                as_attachment=True,
                download_name=f"{analysis_id}_report.json"
            )
        else:
            # Return JSON
            with open(report_path, 'r') as f:
                report = json.load(f)
            return jsonify(report), 200
            
    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve report',
            'message': str(e)
        }), 500


@app.route('/reports', methods=['GET'])
def list_reports():
    """
    List all available reports
    
    Response:
        - List of analysis IDs and metadata
    """
    try:
        reports = []
        
        for report_file in LOGS_FOLDER.glob("*_report.json"):
            try:
                # Extract analysis ID from filename
                analysis_id = report_file.stem.replace('_report', '')
                
                # Get file metadata
                stat = report_file.stat()
                
                # Try to load report summary
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    summary = report.get('summary', {})
                    metadata = report.get('metadata', {})
                
                reports.append({
                    'analysis_id': analysis_id,
                    'filename': metadata.get('target', 'unknown'),
                    'timestamp': metadata.get('timestamp', datetime.fromtimestamp(stat.st_ctime).isoformat()),
                    'size': stat.st_size,
                    'summary': summary
                })
            except:
                pass
        
        # Sort by timestamp (newest first)
        reports.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'count': len(reports),
            'reports': reports
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to list reports',
            'message': str(e)
        }), 500


@app.route('/config', methods=['GET', 'POST'])
def manage_config():
    """
    Get or update server configuration
    
    GET: Returns current configuration
    POST: Updates configuration
    """
    if request.method == 'GET':
        return jsonify(config), 200
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            if 'default_duration' in data:
                config['default_duration'] = int(data['default_duration'])
            
            if 'max_duration' in data:
                config['max_duration'] = int(data['max_duration'])
            
            if 'auto_cleanup' in data:
                config['auto_cleanup'] = bool(data['auto_cleanup'])
            
            return jsonify({
                'status': 'updated',
                'config': config
            }), 200
            
        except Exception as e:
            return jsonify({
                'error': 'Failed to update config',
                'message': str(e)
            }), 400


@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        'service': 'Windows Sandbox API Server',
        'version': '1.0',
        'endpoints': {
            'GET /health': 'Health check',
            'POST /analyze': 'Analyze file (upload PE file)',
            'GET /report/<id>': 'Get analysis report',
            'GET /reports': 'List all reports',
            'GET /config': 'Get configuration',
            'POST /config': 'Update configuration'
        },
        'usage': {
            'analyze': 'curl -X POST -F "file=@malware.exe" -F "duration=120" http://VM_IP:5000/analyze',
            'get_report': 'curl http://VM_IP:5000/report/<analysis_id>',
            'download_report': 'curl http://VM_IP:5000/report/<analysis_id>?download=true -o report.json'
        }
    })


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Windows Sandbox API Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on default port
  python windows_sandbox_server.py
  
  # Start on custom port
  python windows_sandbox_server.py --port 8080
  
  # Allow external connections
  python windows_sandbox_server.py --host 0.0.0.0 --port 5000
  
  # Test from host machine
  curl -X POST -F "file=@malware.exe" http://VM_IP:5000/analyze
        """
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0 for all interfaces)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to listen on (default: 5000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=120,
        help='Default analysis duration in seconds (default: 120)'
    )
    
    args = parser.parse_args()
    
    # Update config
    config['default_duration'] = args.duration
    
    # Check if running on Windows
    if sys.platform != 'win32':
        print(f"❌ Error: This server must run on Windows!")
        print(f"   Current OS: {sys.platform}")
        sys.exit(1)
    
    # Print startup info
    print(f"{'='*60}")
    print(f"🪟 Windows Sandbox API Server")
    print(f"{'='*60}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Default Duration: {config['default_duration']} seconds")
    print(f"Upload Folder: {UPLOAD_FOLDER.absolute()}")
    print(f"Logs Folder: {LOGS_FOLDER.absolute()}")
    print(f"{'='*60}")
    print(f"\n✅ Server starting...")
    print(f"\n📡 API Endpoints:")
    print(f"   Health Check: http://{args.host}:{args.port}/health")
    print(f"   Analyze File: http://{args.host}:{args.port}/analyze")
    print(f"   List Reports: http://{args.host}:{args.port}/reports")
    print(f"\n💡 Test from host machine:")
    print(f'   curl -X POST -F "file=@malware.exe" http://VM_IP:{args.port}/analyze')
    print(f"\n⚠️  Make sure Windows Firewall allows port {args.port}!")
    print(f"{'='*60}\n")
    
    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()

