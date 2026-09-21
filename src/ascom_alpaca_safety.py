"""
ASCOM Alpaca Safety Monitor Server
Provides a REST API compatible with ASCOM Alpaca for safety monitoring
"""

import json
import threading
import socket
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import logging
import logging.handlers
import uuid

# Seconds between safety-status refreshes.
UPDATE_INTERVAL_SECONDS = 30

# A roof classification older than this is not trusted for the IsSafe flag.
MAX_STATUS_AGE_SECONDS = 180

# Reported alongside IsSafe=False while no client has connected the device.
NOT_CONNECTED_MESSAGE = "Device is not connected"

# The server log. Clients poll several times a minute for months on end, so the
# file is size-capped and rotated, and per-request detail is only logged at DEBUG.
LOG_FILE = "ascom_alpaca_safety.log"
LOG_LEVEL = logging.INFO
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3

# ASCOM error numbers. Alpaca requires driver errors in 0x400-0xFFF; the server
# used to report 1 for everything, which clients cannot map to an ASCOM exception.
ERROR_NOT_IMPLEMENTED = 0x400
ERROR_ACTION_NOT_IMPLEMENTED = 0x40C
ERROR_UNSPECIFIED = 0x4FF

# Alpaca transaction IDs are uint32.
_MAX_TRANSACTION_ID = 2 ** 32 - 1


def _lookup_case_insensitive(mapping, name):
    """Return mapping[name], matching the key without regard to case, or None.

    The Alpaca spec makes parameter names case-insensitive, so a client sending
    ``clienttransactionid`` or ``connected`` must be understood.
    """
    if not mapping:
        return None
    if name in mapping:
        return mapping.get(name)
    wanted = name.lower()
    for key in mapping.keys():
        if isinstance(key, str) and key.lower() == wanted:
            return mapping.get(key)
    return None


def parse_bool(value):
    """Parse an Alpaca boolean strictly ("True"/"False", any case).

    Raises ValueError for anything else. The lenient parser this replaces read
    every unrecognised value - a typo, "maybe" - as False, so a garbled
    Connected=... silently disconnected the device.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == 'true':
        return True
    if text == 'false':
        return False
    raise ValueError(f"'{value}' is not a valid boolean (expected True or False)")


def compute_safety(status, sun_safe, age_seconds, max_age_seconds=MAX_STATUS_AGE_SECONDS):
    """Decide the ASCOM IsSafe flag from the latest roof classification.

    Returns ``(is_safe, error_message)``. Conditions are reported safe only when
    the roof is known to be OPEN, that knowledge is recent, and the sun is below
    the configured threshold. Every unknown resolves to unsafe: an ASCOM client
    acts on this flag, so a missing or stale answer must never read as "safe".
    """
    if status is None:
        if age_seconds is None:
            return False, "No roof status available yet"
        return False, f"Roof status is stale ({age_seconds:.0f}s old)"
    if age_seconds is None:
        return False, "Roof status age unavailable"
    if age_seconds > max_age_seconds:
        return False, f"Roof status is stale ({age_seconds:.0f}s old)"
    if status != "OPEN":
        return False, ""
    if not sun_safe:
        return False, ""
    return True, ""


class AscomAlpacaSafetyMonitor:
    """ASCOM Alpaca Safety Monitor implementation"""

    def __init__(self, port=11111, device_number=0, roof_classifier_app=None, unique_id=None,
                 start_background=True):
        self.port = port
        self.device_number = device_number
        self.roof_classifier_app = roof_classifier_app

        # Device properties
        # Use the caller-supplied persistent UniqueID when provided so NINA can
        # reconnect to the same device after a restart/reboot. Fall back to a
        # random UUID only when no stable id was given (e.g. standalone test mode).
        self.device_id = unique_id if unique_id else str(uuid.uuid4())
        self.device_name = "Synthetic Roof Safety Monitor"
        self.device_description = "Safety monitor based on roof image classification"
        self.device_version = "1.0.0"
        self.driver_version = "1.0.0"
        
        # Safety monitor state. IsSafe starts False: a client that connects and
        # polls before the first classification must not be told conditions are
        # safe on the strength of a default value.
        self.connected = False
        self.is_safe = False
        self.last_update = datetime.now(timezone.utc)
        self.last_error = "No roof status available yet"
        
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

        # Signals both background threads to exit; also lets tests construct the
        # server without starting anything.
        self._stop_event = threading.Event()
        # ServerTransactionID must be unique per response, and Flask serves
        # requests from several threads.
        self._transaction_lock = threading.Lock()
        self._server_transaction_id = 0
        self.update_thread = None

        if start_background:
            if self.discovery_enabled:
                self.start_discovery_responder()
            self.update_thread = threading.Thread(target=self.update_safety_status, daemon=True)
            self.update_thread.start()
        
    def setup_logging(self):
        """Setup logging for the ASCOM server"""
        self.logger = logging.getLogger('AscomAlpacaSafetyMonitor')
        # DEBUG logged every request with its full headers, several lines per
        # poll, into a file that was never rotated. Set LOG_LEVEL to
        # logging.DEBUG when troubleshooting a client.
        self.logger.setLevel(LOG_LEVEL)

        # The logger is module-global, so a second server instance would otherwise
        # attach a second set of handlers and double every log line.
        if self.logger.handlers:
            return

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8")
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
        while self.discovery_socket and not self._stop_event.is_set():
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
                    
            except OSError as e:
                # The socket was closed (stop requested), or the OS handed us an
                # unrecoverable error - either way there is nothing left to serve.
                if self.discovery_socket and not self._stop_event.is_set():
                    self.logger.error(f"Discovery socket error, responder stopping: {e}")
                break
            except Exception as e:
                # A malformed packet or a serialisation failure must not take the
                # responder down for the rest of the session: NINA would then never
                # find the device again without a restart.
                self.logger.error(f"Error handling discovery request, continuing: {e}")
                continue
                
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
            data = request.get_json(silent=True)
            if isinstance(data, dict):
                value = _lookup_case_insensitive(data, param_name)

        # Try form data (ASCOM standard)
        if value is None and request.form:
            value = _lookup_case_insensitive(request.form, param_name)

        # Try query parameters
        if value is None:
            value = _lookup_case_insensitive(request.args, param_name)

        # Convert type if needed
        if value is not None:
            if param_type == bool:
                return parse_bool(value)
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
        
        response['ClientTransactionID'] = self._client_transaction_id()

        # Add server transaction ID (incremental, unique across worker threads)
        with self._transaction_lock:
            self._server_transaction_id += 1
            response['ServerTransactionID'] = self._server_transaction_id
        
        return jsonify(response)
    
    def _client_transaction_id(self):
        """The request's ClientTransactionID, or 0 when absent or invalid.

        The spec says to return 0 in that case. A non-numeric value used to raise
        here - and again inside the error handler - so the client got a bare
        HTTP 500 instead of an answer.
        """
        try:
            value = self.get_request_parameter('ClientTransactionID', int)
        except (TypeError, ValueError):
            return 0
        if value is None or not 0 <= value <= _MAX_TRANSACTION_ID:
            return 0
        return value

    def setup_routes(self):
        """Setup Flask routes for ASCOM Alpaca API"""
        
        # Add request logging middleware
        @self.app.before_request
        def log_request():
            # Skip building the dumps below unless someone is going to read them.
            if not self.logger.isEnabledFor(logging.DEBUG):
                return
            self.logger.debug(f"Request: {request.method} {request.path}")
            self.logger.debug(f"Content-Type: {request.content_type}")
            self.logger.debug(f"Headers: {dict(request.headers)}")
            if request.is_json:
                self.logger.debug(f"JSON Data: {request.get_json(silent=True)}")
            elif request.form:
                self.logger.debug(f"Form Data: {dict(request.form)}")
            elif request.args:
                self.logger.debug(f"Query Args: {dict(request.args)}")
        
        # Add error handler
        @self.app.errorhandler(Exception)
        def handle_error(error):
            if isinstance(error, HTTPException):
                # 404, 405 and friends keep their own status; turning every one
                # of them into a 500 hid a wrong method or URL behind a "server
                # error".
                return error.description or error.name, error.code
            self.logger.error(f"Unhandled error: {error}")
            return self.get_ascom_response(None, ERROR_UNSPECIFIED, str(error)), 500
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
            is_safe, reason = self.reported_safety()
            safe_text = "Safe" if is_safe else "Unsafe"
            if reason and not is_safe:
                safe_text += f" ({reason})"
            html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>Synthetic Roof Safety Monitor</title>
                <style>
                    :root {{ color-scheme: light dark; --muted: #666; --line: #d8d8d8; }}
                    @media (prefers-color-scheme: dark) {{ :root {{ --muted: #9a9a9a; --line: #3a3a3a; }} }}
                    body {{ font: 15px/1.5 "Segoe UI", system-ui, sans-serif; max-width: 720px;
                            margin: 40px auto; padding: 0 20px; }}
                    h1 {{ font-size: 22px; margin: 0 0 4px; }}
                    h2 {{ font-size: 15px; margin: 32px 0 8px; }}
                    .sub {{ color: var(--muted); margin: 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 0; text-align: left;
                              vertical-align: top; }}
                    th {{ width: 40%; font-weight: normal; color: var(--muted); }}
                    code {{ font-family: Consolas, ui-monospace, monospace; }}
                    ol, ul {{ padding-left: 20px; }}
                </style>
            </head>
            <body>
                <h1>Synthetic Roof Safety Monitor</h1>
                <p class="sub">ASCOM Alpaca safety monitor, driver {self.driver_version}</p>

                <h2>Current status</h2>
                <table>
                    <tr><th>Client connected</th><td>{"Yes" if self.connected else "No"}</td></tr>
                    <tr><th>Reported to clients</th><td>{safe_text}</td></tr>
                    <tr><th>Last updated</th><td>{self.last_update.strftime("%Y-%m-%d %H:%M:%S")} UTC</td></tr>
                </table>

                <h2>Device</h2>
                <table>
                    <tr><th>Name</th><td>{self.device_name}</td></tr>
                    <tr><th>Type</th><td>SafetyMonitor</td></tr>
                    <tr><th>Device number</th><td>{self.device_number}</td></tr>
                    <tr><th>Port</th><td>{self.port}</td></tr>
                    <tr><th>API version</th><td>1</td></tr>
                </table>

                <h2>Connecting from NINA</h2>
                <ol>
                    <li>Open Equipment, then Safety Monitor.</li>
                    <li>Pick this device from the discovered Alpaca devices, or add it by hand
                        with host <code>localhost</code> (or this computer's address),
                        port <code>{self.port}</code> and device number <code>{self.device_number}</code>.</li>
                    <li>Click Connect.</li>
                </ol>

                <h2>API endpoints</h2>
                <ul>
                        <li><a href="/management/apiversions">/management/apiversions</a></li>
                        <li><a href="/management/v1/description">/management/v1/description</a></li>
                        <li><a href="/management/v1/configureddevices">/management/v1/configureddevices</a></li>
                        <li><a href="/api/v1/safetymonitor/{self.device_number}/issafe">/api/v1/safetymonitor/{self.device_number}/issafe</a></li>
                        <li><a href="/api/v1/safetymonitor/{self.device_number}/status">/api/v1/safetymonitor/{self.device_number}/status</a></li>
                </ul>
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
                # Alpaca: a missing or malformed parameter is an HTTP 400 with a
                # plain-text reason, and must leave the connection state alone.
                try:
                    connected_value = self.get_request_parameter('Connected', bool)
                except ValueError as e:
                    self.logger.warning(f"Rejected Connected value: {e}")
                    return f"Invalid Connected value: {e}", 400
                if connected_value is None:
                    self.logger.warning("No Connected parameter found in request")
                    return "Missing parameter: Connected", 400

                try:
                    self.connected = connected_value
                    if self.connected:
                        self.logger.info("ASCOM client connected")
                        # Refresh now rather than leaving the client to read a
                        # flag that is up to one update interval out of date.
                        self.refresh_safety_status()
                    else:
                        # Refreshes stop while disconnected, so drop the flag
                        # now instead of freezing its last value.
                        self.is_safe = False
                        self.last_error = NOT_CONNECTED_MESSAGE
                        self.logger.info("ASCOM client disconnected")

                    return self.get_ascom_response(None)
                except Exception as e:
                    self.logger.error(f"Error in connected endpoint: {e}")
                    return self.get_ascom_response(None, ERROR_UNSPECIFIED, str(e))
                    
        @self.app.route(f'{device_base}/issafe', methods=['GET'])
        def is_safe():
            """Return current safety status"""
            safe, error = self.reported_safety()
            return self.get_ascom_response(safe, 0, error)
            
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
                return self.get_ascom_response(
                    "", ERROR_ACTION_NOT_IMPLEMENTED, f"Action '{action_name}' is not supported")
            except Exception as e:
                return self.get_ascom_response("", ERROR_UNSPECIFIED, str(e))
        
        @self.app.route(f'{device_base}/commandblind', methods=['PUT'])
        def command_blind():
            """Execute blind command"""
            try:
                command = self.get_request_parameter('Command')
                raw = self.get_request_parameter('Raw', bool)
                
                self.logger.info(f"Blind command: {command}, Raw: {raw}")
                
                # No blind commands supported
                return self.get_ascom_response(
                    None, ERROR_NOT_IMPLEMENTED, f"Command '{command}' is not supported")
            except Exception as e:
                return self.get_ascom_response(None, ERROR_UNSPECIFIED, str(e))
        
        @self.app.route(f'{device_base}/commandbool', methods=['PUT'])
        def command_bool():
            """Execute command returning boolean"""
            try:
                command = self.get_request_parameter('Command')
                raw = self.get_request_parameter('Raw', bool)
                
                self.logger.info(f"Bool command: {command}, Raw: {raw}")
                
                # No bool commands supported
                return self.get_ascom_response(
                    False, ERROR_NOT_IMPLEMENTED, f"Command '{command}' is not supported")
            except Exception as e:
                return self.get_ascom_response(False, ERROR_UNSPECIFIED, str(e))
        
        @self.app.route(f'{device_base}/commandstring', methods=['PUT'])
        def command_string():
            """Execute command returning string"""
            try:
                command = self.get_request_parameter('Command')
                raw = self.get_request_parameter('Raw', bool)
                
                self.logger.info(f"String command: {command}, Raw: {raw}")
                
                # No string commands supported
                return self.get_ascom_response(
                    "", ERROR_NOT_IMPLEMENTED, f"Command '{command}' is not supported")
            except Exception as e:
                return self.get_ascom_response("", ERROR_UNSPECIFIED, str(e))
            
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
                    # Report the most recent classification rather than starting a
                    # new one: this endpoint is a read, and classifying here would
                    # rewrite the roof status file on an arbitrary HTTP request.
                    status, _age = self.roof_classifier_app.get_cached_status()
                    if status:
                        roof_status = status

                    angle = self.roof_classifier_app.calculate_sun_angle()
                    if angle is not None:
                        sun_angle = f"{angle:.1f}°"
                except Exception as e:
                    self.logger.warning(f"Error getting roof status: {e}")
            
            safe, error = self.reported_safety()
            return self.get_ascom_response({
                'IsSafe': safe,
                'RoofStatus': roof_status,
                'SunAngle': sun_angle,
                'LastUpdate': self.last_update.isoformat(),
                'LastError': error
            })
        
        # Catch-all route for debugging unknown requests
        @self.app.route('/api/v1/safetymonitor/<int:device_num>/<path:endpoint>', methods=['GET', 'PUT', 'POST'])
        def catch_all_device(device_num, endpoint):
            """Catch-all for unknown device endpoints"""
            self.logger.warning(f"Unknown device endpoint: {request.method} /api/v1/safetymonitor/{device_num}/{endpoint}")
            self.logger.warning(f"Request data: JSON={request.get_json(silent=True)}, Form={dict(request.form)}, Args={dict(request.args)}")
            return self.get_ascom_response(None, ERROR_NOT_IMPLEMENTED, f"Unknown endpoint: {endpoint}")
        
        @self.app.route('/<path:path>', methods=['GET', 'PUT', 'POST'])
        def catch_all(path):
            """Catch-all for any unknown endpoints"""
            if not path.startswith('setup'):  # Don't log setup page requests
                self.logger.warning(f"Unknown endpoint: {request.method} /{path}")
            return self.get_ascom_response(None, ERROR_NOT_IMPLEMENTED, f"Unknown endpoint: /{path}")
            
    def reported_safety(self):
        """Return ``(is_safe, error_message)`` as clients should see it.

        The ASCOM SafetyMonitor contract is that IsSafe is False while the device
        is not connected. The stored flag is only refreshed while connected, so
        without this gate a client polling IsSafe without connecting first (or
        after another client disconnected) read whatever value was last computed
        - possibly True from hours earlier.
        """
        if not self.connected:
            return False, NOT_CONNECTED_MESSAGE
        return self.is_safe, self.last_error

    def refresh_safety_status(self):
        """Recompute IsSafe once from the classifier's latest result."""
        try:
            if not self.roof_classifier_app:
                # Standalone/misconfigured: nothing to base a judgement on, so fail closed.
                self.is_safe = False
                self.last_error = "No roof classifier attached"
                self.last_update = datetime.now(timezone.utc)
                return

            if not self.connected:
                # No client is listening; leave the flag alone rather than
                # publishing an optimistic value a client might pick up on connect.
                return

            # Ask for the classification the monitor loop already produced. This
            # thread used to run its own classify_latest_png(), which raced the
            # monitor thread for the roof status file and the shared toggle and
            # disagreement state.
            status, age = self.roof_classifier_app.get_cached_status(MAX_STATUS_AGE_SECONDS)
            sun_safe = self.roof_classifier_app.is_sun_safe_for_open()

            self.is_safe, self.last_error = compute_safety(
                status, sun_safe, age, MAX_STATUS_AGE_SECONDS)
            self.last_update = datetime.now(timezone.utc)

            if self.last_error:
                self.logger.warning(f"Reporting unsafe: {self.last_error}")
            else:
                self.logger.debug(
                    f"Safety status updated: Safe={self.is_safe}, Roof={status}, Sun safe={sun_safe}")

        except Exception as e:
            self.is_safe = False
            self.last_error = f"Error updating safety status: {str(e)}"
            self.logger.error(f"Error updating safety status: {e}")

    def update_safety_status(self):
        """Background thread to update safety status"""
        while not self._stop_event.is_set():
            self.refresh_safety_status()
            self._stop_event.wait(UPDATE_INTERVAL_SECONDS)
            
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

        # Signal the update thread and the discovery responder to exit
        self._stop_event.set()

        # Stop discovery responder
        self.stop_discovery_responder()
        
        # Flask doesn't have a built-in way to stop, so we'll use shutdown
        # This would typically be called from a signal handler
        pass

if __name__ == "__main__":
    # Standalone test mode
    monitor = AscomAlpacaSafetyMonitor(port=11111)
    monitor.run()
