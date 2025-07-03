"""
ASCOM Alpaca Safety Monitor Server
Provides a REST API compatible with ASCOM Alpaca for safety monitoring
"""

import json
import threading
import time
import socket
import struct
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import os
import uuid

class AscomAlpacaSafetyMonitor:
    """ASCOM Alpaca Safety Monitor implementation"""
    
    def __init__(self, port=11111, device_number=0, roof_classifier_app=None):
        self.port = port
        self.device_number = device_number
        self.roof_classifier_app = roof_classifier_app
        
        # Device properties
        self.device_id = str(uuid.uuid4())
        self.device_name = "Synthetic Roof Safety Monitor"
        self.device_description = "Safety monitor based on roof image classification"
        self.device_version = "1.0.0"
        self.driver_version = "1.0.0"
        
        # Safety monitor state
        self.connected = False
        self.is_safe = True
        self.last_update = datetime.now(timezone.utc)
        self.last_error = ""
        
        # Discovery settings
        self.discovery_enabled = True
        self.discovery_port = 32227  # Standard ASCOM discovery port
        self.discovery_socket = None
        
        # Create Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Setup logging
        self.setup_logging()
        
        # Setup routes
        self.setup_routes()
        
        # Start discovery responder
        if self.discovery_enabled:
            self.start_discovery_responder()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self.update_safety_status, daemon=True)
        self.update_thread.start()
        
    def setup_logging(self):
        """Setup logging for the ASCOM server"""
        self.logger = logging.getLogger('AscomAlpacaSafetyMonitor')
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG for troubleshooting
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        try:
            file_handler = logging.FileHandler('ascom_alpaca_safety.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.warning(f"Could not setup file logging: {e}")
            
    def start_discovery_responder(self):
        """Start the ASCOM Alpaca discovery responder"""
        try:
            self.discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.discovery_socket.bind(('', self.discovery_port))
            
            # Start discovery thread
            discovery_thread = threading.Thread(target=self.discovery_responder, daemon=True)
            discovery_thread.start()
            
            self.logger.info(f"ASCOM discovery responder started on port {self.discovery_port}")
            
        except Exception as e:
            self.logger.warning(f"Could not start discovery responder: {e}")
            self.discovery_socket = None
            
    def discovery_responder(self):
        """Handle ASCOM discovery requests"""
        while self.discovery_socket:
            try:
                data, addr = self.discovery_socket.recvfrom(1024)
                
                # ASCOM discovery packet format: "alpacadiscovery1"
                if data.startswith(b"alpacadiscovery1"):
                    self.logger.debug(f"Discovery request from {addr}")
                    
                    # Create discovery response
                    response = {
                        "AlpacaPort": self.port,
                        "ServerName": "Synthetic Roof Safety Monitor",
                        "Manufacturer": "Synthetic Roof Project",
                        "ManufacturerVersion": "1.0.0",
                        "Location": "Observatory"
                    }
                    
                    # Send JSON response
                    response_data = json.dumps(response).encode('utf-8')
                    self.discovery_socket.sendto(response_data, addr)
                    
                    self.logger.debug(f"Discovery response sent to {addr}")
                    
            except Exception as e:
                if self.discovery_socket:  # Only log if we're still supposed to be running
                    self.logger.error(f"Error in discovery responder: {e}")
                break
                
    def stop_discovery_responder(self):
        """Stop the discovery responder"""
        if self.discovery_socket:
            try:
                self.discovery_socket.close()
                self.discovery_socket = None
                self.logger.info("Discovery responder stopped")
            except Exception as e:
                self.logger.warning(f"Error stopping discovery responder: {e}")
            
    def get_request_parameter(self, param_name, param_type=str):
        """
        Get parameter from request, supporting both JSON and form data
        ASCOM Alpaca standard uses form-encoded data
        """
        value = None
        
        # Try JSON first
        if request.is_json:
            data = request.get_json()
            if data and param_name in data:
                value = data[param_name]
        
        # Try form data (ASCOM standard)
        if value is None and request.form:
            value = request.form.get(param_name)
        
        # Try query parameters
        if value is None:
            value = request.args.get(param_name)
        
        # Convert type if needed
        if value is not None:
            if param_type == bool:
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ('true', '1', 'yes', 'on')
            elif param_type == int:
                return int(value)
            elif param_type == float:
                return float(value)
            else:
                return str(value)
        
        return None
    
    def get_ascom_response(self, value, error_number=0, error_message=''):
        """
        Create standard ASCOM response with optional client tracking
        """
        response = {
            'Value': value,
            'ErrorNumber': error_number,
            'ErrorMessage': error_message
        }
        
        # Add client transaction ID if provided
        client_transaction_id = self.get_request_parameter('ClientTransactionID', int)
        if client_transaction_id is not None:
            response['ClientTransactionID'] = client_transaction_id
        
        # Add server transaction ID (incremental)
        if not hasattr(self, '_server_transaction_id'):
            self._server_transaction_id = 0
        self._server_transaction_id += 1
        response['ServerTransactionID'] = self._server_transaction_id
        
        return jsonify(response)
    
    def setup_routes(self):
        """Setup Flask routes for ASCOM Alpaca API"""
        
        # Add request logging middleware
        @self.app.before_request
        def log_request():
            self.logger.debug(f"Request: {request.method} {request.path}")
            self.logger.debug(f"Content-Type: {request.content_type}")
            self.logger.debug(f"Headers: {dict(request.headers)}")
            if request.is_json:
                self.logger.debug(f"JSON Data: {request.get_json()}")
            elif request.form:
                self.logger.debug(f"Form Data: {dict(request.form)}")
            elif request.args:
                self.logger.debug(f"Query Args: {dict(request.args)}")
        
        # Add error handler
        @self.app.errorhandler(Exception)
        def handle_error(error):
            self.logger.error(f"Unhandled error: {error}")
            return self.get_ascom_response(None, 1, str(error)), 500
        @self.app.route('/management/apiversions', methods=['GET'])
        def api_versions():
            """Return supported API versions"""
            return self.get_ascom_response([1])
            
        @self.app.route('/management/v1/description', methods=['GET'])
        def management_description():
            """Return server description"""
            return self.get_ascom_response({
                'ServerName': 'Synthetic Roof Safety Monitor Server',
                'Manufacturer': 'Synthetic Roof Project',
                'Version': '1.0.0',
                'Location': 'Observatory'
            })
            
        @self.app.route('/management/v1/configureddevices', methods=['GET'])
        def configured_devices():
            """Return list of configured devices"""
            return self.get_ascom_response([{
                'DeviceName': self.device_name,
                'DeviceType': 'SafetyMonitor',
                'DeviceNumber': self.device_number,
                'UniqueID': self.device_id
            }])
            
        @self.app.route('/setup', methods=['GET'])
        def setup_page():
            """Return a simple setup page for web browsers"""
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Synthetic Roof Safety Monitor Setup</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .info {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                    .status {{ background: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Synthetic Roof Safety Monitor</h1>
                <div class="info">
                    <h3>Device Information</h3>
                    <table>
                        <tr><th>Property</th><th>Value</th></tr>
                        <tr><td>Device Name</td><td>{self.device_name}</td></tr>
                        <tr><td>Device Type</td><td>SafetyMonitor</td></tr>
                        <tr><td>Device Number</td><td>{self.device_number}</td></tr>
                        <tr><td>Port</td><td>{self.port}</td></tr>
                        <tr><td>API Version</td><td>1</td></tr>
                        <tr><td>Driver Version</td><td>{self.driver_version}</td></tr>
                    </table>
                </div>
                
                <div class="status">
                    <h3>Current Status</h3>
                    <p><strong>Connected:</strong> {self.connected}</p>
                    <p><strong>Safe:</strong> {self.is_safe}</p>
                    <p><strong>Last Update:</strong> {self.last_update.isoformat()}</p>
                </div>
                
                <div class="info">
                    <h3>NINA Setup Instructions</h3>
                    <ol>
                        <li>In NINA, go to Equipment → Safety Monitor</li>
                        <li>Choose "ASCOM" as the safety monitor type</li>
                        <li>Select "Alpaca Safety Monitor" or use discovery</li>
                        <li>Set IP address to: <strong>localhost</strong> (or this computer's IP)</li>
                        <li>Set port to: <strong>{self.port}</strong></li>
                        <li>Set device number to: <strong>{self.device_number}</strong></li>
                        <li>Click "Connect"</li>
                    </ol>
                </div>
                
                <div class="info">
                    <h3>API Endpoints</h3>
                    <ul>
                        <li><a href="/management/apiversions">/management/apiversions</a></li>
                        <li><a href="/management/v1/description">/management/v1/description</a></li>
                        <li><a href="/management/v1/configureddevices">/management/v1/configureddevices</a></li>
                        <li><a href="/api/v1/safetymonitor/{self.device_number}/issafe">/api/v1/safetymonitor/{self.device_number}/issafe</a></li>
                        <li><a href="/api/v1/safetymonitor/{self.device_number}/status">/api/v1/safetymonitor/{self.device_number}/status</a></li>
                    </ul>
                </div>
            </body>
            </html>
            """
            return html
            
        # Device API
        device_base = f'/api/v1/safetymonitor/{self.device_number}'
        
        @self.app.route(f'{device_base}/connected', methods=['GET', 'PUT'])
        def connected():
            """Get or set connection state"""
            if request.method == 'GET':
                return self.get_ascom_response(self.connected)
            else:
                try:
                    # Get Connected parameter using ASCOM standard
                    connected_value = self.get_request_parameter('Connected', bool)
                    
                    if connected_value is not None:
                        self.connected = connected_value
                        if self.connected:
                            self.logger.info("ASCOM client connected")
                        else:
                            self.logger.info("ASCOM client disconnected")
                    else:
                        self.logger.warning("No Connected parameter found in request")
                    
                    return self.get_ascom_response(None)
                except Exception as e:
                    self.logger.error(f"Error in connected endpoint: {e}")
                    return self.get_ascom_response(None, 1, str(e))
                    
        @self.app.route(f'{device_base}/issafe', methods=['GET'])
        def is_safe():
            """Return current safety status"""
            return self.get_ascom_response(self.is_safe, 0, self.last_error)
            
        @self.app.route(f'{device_base}/name', methods=['GET'])
        def device_name():
            """Return device name"""
            return self.get_ascom_response(self.device_name)
            
        @self.app.route(f'{device_base}/description', methods=['GET'])
        def device_description():
            """Return device description"""
            return self.get_ascom_response(self.device_description)
            
        @self.app.route(f'{device_base}/driverinfo', methods=['GET'])
        def driver_info():
            """Return driver information"""
            return self.get_ascom_response(f"Synthetic Roof Safety Monitor v{self.driver_version}")
            
        @self.app.route(f'{device_base}/driverversion', methods=['GET'])
        def driver_version():
            """Return driver version"""
            return self.get_ascom_response(self.driver_version)
            
        @self.app.route(f'{device_base}/interfaceversion', methods=['GET'])
        def interface_version():
            """Return interface version"""
            return self.get_ascom_response(1)
            
        @self.app.route(f'{device_base}/supportedactions', methods=['GET'])
        def supported_actions():
            """Return supported actions"""
            return self.get_ascom_response([])
        
        # Common ASCOM device properties
        @self.app.route(f'{device_base}/action', methods=['PUT'])
        def action():
            """Execute device action"""
            try:
                action_name = self.get_request_parameter('Action')
                parameters = self.get_request_parameter('Parameters', str)
                
                self.logger.info(f"Action requested: {action_name} with parameters: {parameters}")
                
                # No actions supported for safety monitor
                return self.get_ascom_response("", 1, f"Action '{action_name}' is not supported")
            except Exception as e:
                return self.get_ascom_response("", 1, str(e))
        
        @self.app.route(f'{device_base}/commandblind', methods=['PUT'])
        def command_blind():
            """Execute blind command"""
            try:
                command = self.get_request_parameter('Command')
                raw = self.get_request_parameter('Raw', bool)
                
                self.logger.info(f"Blind command: {command}, Raw: {raw}")
                
                # No blind commands supported
                return self.get_ascom_response(None, 1, f"Command '{command}' is not supported")
            except Exception as e:
                return self.get_ascom_response(None, 1, str(e))
        
        @self.app.route(f'{device_base}/commandbool', methods=['PUT'])
        def command_bool():
            """Execute command returning boolean"""
            try:
                command = self.get_request_parameter('Command')
                raw = self.get_request_parameter('Raw', bool)
                
                self.logger.info(f"Bool command: {command}, Raw: {raw}")
                
                # No bool commands supported
                return self.get_ascom_response(False, 1, f"Command '{command}' is not supported")
            except Exception as e:
                return self.get_ascom_response(False, 1, str(e))
        
        @self.app.route(f'{device_base}/commandstring', methods=['PUT'])
        def command_string():
            """Execute command returning string"""
            try:
                command = self.get_request_parameter('Command')
                raw = self.get_request_parameter('Raw', bool)
                
                self.logger.info(f"String command: {command}, Raw: {raw}")
                
                # No string commands supported
                return self.get_ascom_response("", 1, f"Command '{command}' is not supported")
            except Exception as e:
                return self.get_ascom_response("", 1, str(e))
            
        # Additional safety monitor specific methods
        @self.app.route(f'{device_base}/lastupdate', methods=['GET'])
        def last_update():
            """Return last update time"""
            return self.get_ascom_response(self.last_update.isoformat())
            
        @self.app.route(f'{device_base}/status', methods=['GET'])
        def status():
            """Return detailed status information"""
            roof_status = "UNKNOWN"
            sun_angle = "N/A"
            
            if self.roof_classifier_app:
                try:
                    # Get current roof status
                    filename, status = self.roof_classifier_app.classify_latest_png()
                    if status:
                        roof_status = status
                    
                    # Get sun angle
                    sun_angle = f"{self.roof_classifier_app.calculate_sun_angle():.1f}°"
                except Exception as e:
                    self.logger.warning(f"Error getting roof status: {e}")
            
            return self.get_ascom_response({
                'IsSafe': self.is_safe,
                'RoofStatus': roof_status,
                'SunAngle': sun_angle,
                'LastUpdate': self.last_update.isoformat(),
                'LastError': self.last_error
            })
        
        # Catch-all route for debugging unknown requests
        @self.app.route('/api/v1/safetymonitor/<int:device_num>/<path:endpoint>', methods=['GET', 'PUT', 'POST'])
        def catch_all_device(device_num, endpoint):
            """Catch-all for unknown device endpoints"""
            self.logger.warning(f"Unknown device endpoint: {request.method} /api/v1/safetymonitor/{device_num}/{endpoint}")
            self.logger.warning(f"Request data: JSON={request.get_json()}, Form={dict(request.form)}, Args={dict(request.args)}")
            return self.get_ascom_response(None, 1, f"Unknown endpoint: {endpoint}")
        
        @self.app.route('/<path:path>', methods=['GET', 'PUT', 'POST'])
        def catch_all(path):
            """Catch-all for any unknown endpoints"""
            if not path.startswith('setup'):  # Don't log setup page requests
                self.logger.warning(f"Unknown endpoint: {request.method} /{path}")
            return self.get_ascom_response(None, 1, f"Unknown endpoint: /{path}")
            
    def update_safety_status(self):
        """Background thread to update safety status"""
        while True:
            try:
                if self.roof_classifier_app and self.connected:
                    # Get current roof status
                    filename, status = self.roof_classifier_app.classify_latest_png()
                    
                    if status:
                        # Safety logic: safe if roof is OPEN and sun conditions are safe
                        roof_open = (status == "OPEN")
                        sun_safe = self.roof_classifier_app.is_sun_safe_for_open()
                        
                        # Safe only if both conditions are met
                        self.is_safe = roof_open and sun_safe
                        
                        self.last_update = datetime.now(timezone.utc)
                        self.last_error = ""
                        
                        self.logger.debug(f"Safety status updated: Safe={self.is_safe}, Roof={status}, Sun safe={sun_safe}")
                        
                    else:
                        self.is_safe = False
                        self.last_error = "Unable to determine roof status"
                        self.logger.warning("Could not determine roof status")
                        
                else:
                    # If not connected or no roof classifier, default to safe
                    self.is_safe = True
                    
            except Exception as e:
                self.is_safe = False
                self.last_error = f"Error updating safety status: {str(e)}"
                self.logger.error(f"Error updating safety status: {e}")
                
            time.sleep(30)  # Update every 30 seconds
            
    def run(self):
        """Start the ASCOM Alpaca server"""
        self.logger.info(f"Starting ASCOM Alpaca Safety Monitor on port {self.port}")
        self.logger.info(f"Device number: {self.device_number}")
        self.logger.info(f"Management API: http://localhost:{self.port}/management/apiversions")
        self.logger.info(f"Device API: http://localhost:{self.port}/api/v1/safetymonitor/{self.device_number}/")
        
        try:
            self.app.run(host='0.0.0.0', port=self.port, debug=False)
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            raise
            
    def stop(self):
        """Stop the ASCOM Alpaca server"""
        self.logger.info("Stopping ASCOM Alpaca Safety Monitor")
        
        # Stop discovery responder
        self.stop_discovery_responder()
        
        # Flask doesn't have a built-in way to stop, so we'll use shutdown
        # This would typically be called from a signal handler
        pass

if __name__ == "__main__":
    # Standalone test mode
    monitor = AscomAlpacaSafetyMonitor(port=11111)
    monitor.run()
