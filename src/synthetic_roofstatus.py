import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
import os
import re
import sys
import cv2
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import threading
import math
import time
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import logging.handlers
import shutil
import urllib.request
import urllib.error
import random
import tempfile
import ephem
import pytz
from roof_io import (
    atomic_write_text,
    format_status_line,
    load_json_with_recovery,
    parse_roof_status,
    write_json_atomic,
)
try:
    from astropy.io import fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False

# ASCOM Alpaca dependencies
try:
    from flask import Flask
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

if FLASK_AVAILABLE:
    from ascom_alpaca_safety import AscomAlpacaSafetyMonitor

IMG_SIZE = 32
SETTINGS_FILE = "roof_classifier_settings.json"

# Size cap for the classifier log file, and how many rotated copies to keep.
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3

# Image display size constants
_CLASSIFY_IMG_MAX_W = 820  # max width in the classify-images window (leaves room for button bar)
_CLASSIFY_IMG_MAX_H = 460  # max height in the classify-images window
_PREVIEW_IMG_MAX_W = 460   # max width of the latest-image preview on the Monitoring tab
_PREVIEW_IMG_MAX_H = 380   # max height (all-sky frames are usually square)

# Colours. Only status text is coloured; everything else uses the platform theme.
COLOR_TEXT = "#1f1f1f"
COLOR_MUTED = "#666666"
COLOR_OK = "#1a7f37"
COLOR_WARN = "#9a5b00"
COLOR_ERROR = "#b42318"
COLOR_VIEWER_BG = "#1c1c1c"   # behind images, so frames are judged against a neutral dark
COLOR_VIEWER_TEXT = "#a0a0a0"
ROOF_STATUS_COLORS = {"OPEN": COLOR_OK, "CLOSED": COLOR_ERROR}

# Settings are written this long after the last change, so typing into a field
# does not rewrite the settings file on every keystroke.
SETTINGS_SAVE_DELAY_MS = 600

# Manual override duration choices shown in the Monitoring tab dropdown
OVERRIDE_DURATIONS = ["1 hour", "4 hours", "Until noon", "Until midnight", "Forever"]

# A cached classification newer than this is reused instead of running the whole
# capture/classify pipeline again. The monitor loop runs every 60s, so an ASCOM
# poll in between is served from the cache.
CLASSIFICATION_CACHE_SECONDS = 30

# A cached classification older than this is treated as unavailable: the ASCOM
# safety monitor must not keep reporting a stale roof state as authoritative.
CLASSIFICATION_MAX_AGE_SECONDS = 180

# Seconds between monitoring passes, and the granularity of the countdown (and of
# noticing a stop request) in between.
MONITOR_INTERVAL_SECONDS = 60
_COUNTDOWN_TICK_SECONDS = 1.0

# When no pass has produced a classification for this long, the status file is
# rewritten as CLOSED rather than left showing the last good (possibly OPEN)
# line indefinitely. Matches the age after which ASCOM stops trusting a result.
FAILSAFE_AFTER_SECONDS = CLASSIFICATION_MAX_AGE_SECONDS

# Default stale-image threshold (minutes) when none is configured. An image that
# has not changed for this long raises the stale notification and is reported
# CLOSED rather than trusted.
DEFAULT_STALE_MINUTES = 10.0

# What to report while the image is stale: keep reporting whatever the model
# makes of the frozen frame (the default), or CLOSED as a fail-safe.
STALE_ACTION_CLOSED = "closed"
STALE_ACTION_KEEP = "keep"
STALE_ACTIONS = (STALE_ACTION_CLOSED, STALE_ACTION_KEEP)
DEFAULT_STALE_ACTION = STALE_ACTION_KEEP

# How long a secondary roof status fetched over HTTP is reused before re-fetching.
# Keeps the UI thread from issuing a network request on every status-label refresh.
_SECONDARY_URL_CACHE_SECONDS = 30

# Twilight angle presets (standard astronomical definitions)
TWILIGHT_PRESETS = {
    "Sunset / Sunrise": "0.0",      # Sun at horizon
    "Civil": "-6.0",       # Civil twilight (brightest stars visible)
    "Nautical": "-12.0",   # Nautical twilight (horizon barely visible)
    "Astronomical": "-18.0" # Astronomical twilight (complete darkness)
}

class RoofClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic RoofStatusFile.txt")
        self.model_path = tk.StringVar()
        self.monitor_path = tk.StringVar()
        self.output_path = tk.StringVar(value="RoofStatusFile.txt")
        
        # New configuration variables
        self.log_enabled = tk.BooleanVar(value=False)
        self.log_path = tk.StringVar(value="roof_classifier.log")
        self.latitude = tk.StringVar(value="40.0")  # Default latitude
        self.longitude = tk.StringVar(value="-74.0")  # Default longitude
        self.sun_angle_threshold = tk.StringVar(value="-17.0")  # Default sun angle threshold
        self.secondary_source_enabled = tk.BooleanVar(value=False)
        self.secondary_source_path = tk.StringVar()
        self.twilight_preset_var = tk.StringVar(value="Custom")
        
        # ASCOM Alpaca configuration
        self.ascom_enabled = tk.BooleanVar(value=False)
        self.ascom_port = tk.StringVar(value="11111")
        self.ascom_device_number = tk.StringVar(value="0")
        # Stable UniqueID so NINA can reconnect to the same device after a reboot.
        # Generated once and persisted; do NOT regenerate per launch.
        self.ascom_unique_id = ""

        # Auto-start options
        self.auto_start_monitoring = tk.BooleanVar(value=False)
        self.auto_start_ascom = tk.BooleanVar(value=False)

        # Frame capture options
        # Save the frame whenever the model's reported status toggles (OPEN<->CLOSED),
        # so a mis-classified transition can be reviewed/labelled manually later.
        self.save_on_toggle_enabled = tk.BooleanVar(value=False)
        # Save the first frame of each episode where the model disagrees with the
        # secondary roof status file.
        self.save_on_disagreement_enabled = tk.BooleanVar(value=False)

        # Camera URL for remote image source
        self.camera_url = tk.StringVar(value="")

        # Manual override of the reported roof status.
        # override_mode is the *staged* radio-button selection; the committed override
        # lives in self.override_active / self.override_expiry.
        self.override_mode = tk.StringVar(value="AUTO")          # AUTO | OPEN | CLOSED
        self.override_duration = tk.StringVar(value="1 hour")
        self.override_active = None      # None, "OPEN" or "CLOSED" — currently in force
        self.override_expiry = None      # naive local datetime, or None for "Forever"

        # Latest-image preview state (Monitoring tab)
        self.preview_enabled = tk.BooleanVar(value=True)
        self._preview_on = True          # plain-bool mirror, readable from the monitor thread
        self._preview_tk_img = None      # keep a reference so Tk does not garbage-collect it
        self._preview_tmp_path = None    # scaled PNG currently displayed
        self._preview_busy = False       # guards manual refresh from stacking up

        # Cached secondary roof status when the source is an HTTP URL
        self._secondary_cache = None     # (source, status, mod_time, fetched_at) or None

        # Last configuration snapshot handed to a worker thread (see _get_monitor_config)
        self._last_monitor_config = None

        # Classification is shared mutable state (image hash, toggle baseline,
        # disagreement episode, preview, the status file itself), and both the
        # monitor thread and the ASCOM server thread ask for it. Serialise it, and
        # cache the result so the second caller reuses the first one's work.
        self._classify_lock = threading.RLock()
        self._last_classification = None  # (filename, status, datetime UTC) or None

        # Image hash tracking state
        self.last_image_hash = None
        self.last_new_hash_time = None  # datetime (UTC) when hash last changed

        # State tracking for roof open/close transition notifications
        self.previous_status = None
        self._last_stale_notification_time = None
        self._last_heartbeat_time = None

        # State tracking for frame capture / disagreement detection
        self.previous_classified_status = None  # last final status seen by the monitor loop
        self._in_disagreement = False           # currently in a model/secondary disagreement episode

        # Notification settings
        self.notif_stale_enabled = tk.BooleanVar(value=False)
        self.notif_stale_minutes = tk.StringVar(value="10")
        self.notif_stale_url = tk.StringVar(value="")
        self.stale_image_action = tk.StringVar(value=DEFAULT_STALE_ACTION)
        self.notif_open_enabled = tk.BooleanVar(value=False)
        self.notif_open_url = tk.StringVar(value="")
        self.notif_closed_enabled = tk.BooleanVar(value=False)
        self.notif_closed_url = tk.StringVar(value="")
        self.notif_heartbeat_enabled = tk.BooleanVar(value=False)
        self.notif_heartbeat_minutes = tk.StringVar(value="5")
        self.notif_heartbeat_url = tk.StringVar(value="")
        
        # Training set management configuration
        self.training_data_folder = tk.StringVar(value="")
        self.sample_mode_enabled = tk.BooleanVar(value=False)
        self.sample_rate = tk.StringVar(value="0.1")
        self.validation_set_path = tk.StringVar(value="")

        self.model = None
        # Each monitoring run gets its own stop event, so a loop that is still
        # finishing after Stop can never be revived by the next Start (see
        # start_monitoring), and its exit cannot reset the new run's UI.
        self._monitor_stop_event = None
        self.monitoring_active = False
        self._obs_window_after_id = None
        # Fail-safe bookkeeping: when the last pass that produced a classification
        # ran, and whether the status file currently holds a fail-safe line.
        self._last_good_pass_at = None
        self._failsafe_active = False
        self.logger = None
        self.ascom_server = None
        self._startup_model_error = None
        # Why the last reported status differs from the model's raw call, if it does
        self._last_status_reason = ""
        # (status, updated) from the last pass's secondary-source read
        self._last_secondary = None
        # Window layout remembered between sessions
        self._window_geometry = ""
        self._window_zoomed = False
        self._last_tab = None
        self.load_settings()
        self.setup_logging()
        # Before the GUI, so it opens on the right tab and shows the loaded model
        self._try_load_model_from_settings()
        self.setup_gui()
        # Defer auto-start until the main loop is running so the UI is visible first.
        self.root.after(800, self._apply_auto_start)

    def load_settings(self):
        """Load settings from JSON file"""
        def _report_corrupt(backup, exc):
            where = f" It was moved aside to {backup}." if backup else ""
            print(f"Settings file {SETTINGS_FILE} is corrupt ({exc}); "
                  f"falling back to defaults.{where}")

        try:
            settings = load_json_with_recovery(SETTINGS_FILE, on_corrupt=_report_corrupt)
            if settings is not None:
                self.model_path.set(settings.get('model_path', ''))
                self.monitor_path.set(settings.get('monitor_path', ''))
                self.output_path.set(settings.get('output_path', 'RoofStatusFile.txt'))

                # Load new settings
                self.log_enabled.set(settings.get('log_enabled', False))
                self.log_path.set(settings.get('log_path', 'roof_classifier.log'))
                self.latitude.set(settings.get('latitude', '40.0'))
                self.longitude.set(settings.get('longitude', '-74.0'))
                self.sun_angle_threshold.set(settings.get('sun_angle_threshold', '-17.0'))
                self.secondary_source_enabled.set(settings.get('secondary_source_enabled', False))
                self.secondary_source_path.set(settings.get('secondary_source_path', ''))
                self.twilight_preset_var.set(settings.get('twilight_preset', 'Custom'))

                # ASCOM Alpaca settings
                self.ascom_enabled.set(settings.get('ascom_enabled', False))
                self.ascom_port.set(settings.get('ascom_port', '11111'))
                self.ascom_device_number.set(settings.get('ascom_device_number', '0'))
                self.ascom_unique_id = settings.get('ascom_unique_id', '')

                # Auto-start settings
                self.auto_start_monitoring.set(settings.get('auto_start_monitoring', False))
                self.auto_start_ascom.set(settings.get('auto_start_ascom', False))

                # Frame capture settings
                self.save_on_toggle_enabled.set(settings.get('save_on_toggle_enabled', False))
                self.save_on_disagreement_enabled.set(settings.get('save_on_disagreement_enabled', False))

                # Training set management settings
                self.training_data_folder.set(settings.get('training_data_folder', ''))
                self.sample_mode_enabled.set(settings.get('sample_mode_enabled', False))
                self.sample_rate.set(settings.get('sample_rate', '0.1'))
                self.validation_set_path.set(settings.get('validation_set_path', ''))

                # Camera URL
                self.camera_url.set(settings.get('camera_url', ''))

                # Latest-image preview
                self.preview_enabled.set(settings.get('preview_enabled', True))
                self._preview_on = self.preview_enabled.get()

                # Manual override — restored so an override survives a restart
                self.override_duration.set(settings.get('override_duration', '1 hour'))
                saved_override = settings.get('override_active') or None
                saved_expiry = settings.get('override_expiry') or None
                if saved_override in ("OPEN", "CLOSED"):
                    expiry = None
                    expiry_invalid = False
                    if saved_expiry:
                        try:
                            expiry = datetime.fromisoformat(saved_expiry)
                        except (ValueError, TypeError):
                            expiry_invalid = True
                    if expiry_invalid:
                        # A time-limited override with an unreadable expiry must never
                        # be promoted to an indefinite one — discard it instead.
                        print(f"Discarding manual override with invalid expiry: {saved_expiry!r}")
                        saved_override = None
                    elif expiry is not None and expiry <= datetime.now():
                        # Drop an override that expired while the app was closed
                        saved_override = None
                        expiry = None
                    self.override_active = saved_override
                    self.override_expiry = expiry if saved_override else None
                    self.override_mode.set(saved_override or "AUTO")

                # Notification settings
                self.notif_stale_enabled.set(settings.get('notif_stale_enabled', False))
                self.notif_stale_minutes.set(settings.get('notif_stale_minutes', '10'))
                self.notif_stale_url.set(settings.get('notif_stale_url', ''))
                stale_action = settings.get('stale_image_action', DEFAULT_STALE_ACTION)
                self.stale_image_action.set(
                    stale_action if stale_action in STALE_ACTIONS else DEFAULT_STALE_ACTION)
                self.notif_open_enabled.set(settings.get('notif_open_enabled', False))
                self.notif_open_url.set(settings.get('notif_open_url', ''))
                self.notif_closed_enabled.set(settings.get('notif_closed_enabled', False))
                self.notif_closed_url.set(settings.get('notif_closed_url', ''))
                self.notif_heartbeat_enabled.set(settings.get('notif_heartbeat_enabled', False))
                self.notif_heartbeat_minutes.set(settings.get('notif_heartbeat_minutes', '5'))
                self.notif_heartbeat_url.set(settings.get('notif_heartbeat_url', ''))

                # Window layout
                self._window_geometry = settings.get('window_geometry', '')
                self._window_zoomed = bool(settings.get('window_zoomed', False))
                self._last_tab = settings.get('last_tab')
        except Exception as e:
            print(f"Error loading settings: {e}")

    def save_settings(self):
        """Save settings to JSON file"""
        try:
            settings = {
                'model_path': self.model_path.get(),
                'monitor_path': self.monitor_path.get(),
                'output_path': self.output_path.get(),
                'log_enabled': self.log_enabled.get(),
                'log_path': self.log_path.get(),
                'latitude': self.latitude.get(),
                'longitude': self.longitude.get(),
                'sun_angle_threshold': self.sun_angle_threshold.get(),
                'secondary_source_enabled': self.secondary_source_enabled.get(),
                'secondary_source_path': self.secondary_source_path.get(),
                'twilight_preset': self.twilight_preset_var.get(),
                'ascom_enabled': self.ascom_enabled.get(),
                'ascom_port': self.ascom_port.get(),
                'ascom_device_number': self.ascom_device_number.get(),
                'ascom_unique_id': self.ascom_unique_id,
                'auto_start_monitoring': self.auto_start_monitoring.get(),
                'auto_start_ascom': self.auto_start_ascom.get(),
                'save_on_toggle_enabled': self.save_on_toggle_enabled.get(),
                'save_on_disagreement_enabled': self.save_on_disagreement_enabled.get(),
                'training_data_folder': self.training_data_folder.get(),
                'sample_mode_enabled': self.sample_mode_enabled.get(),
                'sample_rate': self.sample_rate.get(),
                'validation_set_path': self.validation_set_path.get(),
                'camera_url': self.camera_url.get(),
                'preview_enabled': self.preview_enabled.get(),
                'override_active': self.override_active or '',
                'override_expiry': self.override_expiry.isoformat() if self.override_expiry else '',
                'override_duration': self.override_duration.get(),
                'notif_stale_enabled': self.notif_stale_enabled.get(),
                'notif_stale_minutes': self.notif_stale_minutes.get(),
                'notif_stale_url': self.notif_stale_url.get(),
                'stale_image_action': self.stale_image_action.get(),
                'notif_open_enabled': self.notif_open_enabled.get(),
                'notif_open_url': self.notif_open_url.get(),
                'notif_closed_enabled': self.notif_closed_enabled.get(),
                'notif_closed_url': self.notif_closed_url.get(),
                'notif_heartbeat_enabled': self.notif_heartbeat_enabled.get(),
                'notif_heartbeat_minutes': self.notif_heartbeat_minutes.get(),
                'notif_heartbeat_url': self.notif_heartbeat_url.get(),
                'window_geometry': getattr(self, '_window_geometry', ''),
                'window_zoomed': getattr(self, '_window_zoomed', False),
                'last_tab': getattr(self, '_last_tab', None),
            }
            # Atomic: a crash part-way through a settings save used to leave an
            # unparseable file, which silently reset every setting - including the
            # persisted ASCOM UniqueID and any active manual override.
            write_json_atomic(SETTINGS_FILE, settings)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def _log_path_conflicts_with_output(self):
        """Return True if the classifier log path resolves to the same file as the roof status output path."""
        return os.path.abspath(self.log_path.get()) == os.path.abspath(self.output_path.get())

    def _warn_log_path_conflict(self, extra_hint=""):
        """Show a warning dialog when the log path conflicts with the roof status output path."""
        hint = f" {extra_hint}" if extra_hint else ""
        messagebox.showwarning(
            "Log Path Conflict",
            f"The classifier log file path is the same as the roof status output file:\n\n"
            f"  {self.output_path.get()}\n\n"
            f"File logging has been disabled to protect the roof status file."
            f"{hint}"
        )

    def setup_logging(self):
        """Setup logging configuration"""
        if hasattr(self, 'logger') and self.logger:
            # Remove and close the existing handlers. Removing alone leaked the
            # log file handle on every Start Monitoring, and on Windows an open
            # handle stops the file from being rotated, moved or deleted.
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                handler.close()
        
        self.logger = logging.getLogger('RoofClassifier')
        self.logger.setLevel(logging.INFO)
        # Prevent classifier log messages from propagating to the root logger,
        # which could cause them to appear in unexpected handlers.
        self.logger.propagate = False
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        if self.log_enabled.get():
            log_path = self.log_path.get()
            # Guard: do not let the classifier log file handler write to the same
            # file as the roof status output file.  If both paths resolve to the
            # same file the diagnostic log entries would corrupt the roof status
            # file format that downstream tools (e.g. ASCOM safety monitors) depend on.
            if self._log_path_conflicts_with_output():
                print(f"Warning: Log file path '{log_path}' is the same as the roof status output file. "
                      f"File logging disabled to protect the roof status file.")
            else:
                try:
                    # File handler
                    # Rotated: the app runs unattended for months and logs every
                    # pass, so an unbounded file eventually fills the disk.
                    file_handler = logging.handlers.RotatingFileHandler(
                        log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT,
                        encoding="utf-8")
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
                except Exception as e:
                    print(f"Error setting up file logging: {e}")
        
        # Console handler (always enabled)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _try_load_model_from_settings(self):
        """Auto-load the model stored in settings so monitoring is ready on startup"""
        path = self.model_path.get()
        if path and os.path.isfile(path):
            try:
                self.model = self._load_model_file(path)
                if self.logger:
                    self.logger.info(f"Auto-loaded model from {path}")
            except Exception as e:
                msg = f"Failed to auto-load model from {path}: {e}"
                if self.logger:
                    self.logger.error(msg)
                # Defer the messagebox until after the main loop starts so the UI is visible
                self.root.after(500, lambda m=msg: messagebox.showerror("Model Load Error", m))

    def _apply_auto_start(self):
        """Honour the auto-start-on-launch options for ASCOM and monitoring."""
        if self.auto_start_ascom.get() and FLASK_AVAILABLE and not self.ascom_server:
            self.start_ascom_server(silent=True)
            self._update_ascom_display()

        if self.auto_start_monitoring.get() and not self.monitoring_active:
            if self.model:
                self.start_monitoring()
            elif self.logger:
                self.logger.warning("Auto-start monitoring skipped: no model loaded.")

    def start_ascom_server(self, silent=False):
        """Start the ASCOM Alpaca server.

        Success is shown in the ASCOM section and the status bar rather than in a
        dialog. When *silent* is True (auto-start on launch) errors are logged
        instead of shown.
        """
        if not FLASK_AVAILABLE:
            if not silent:
                messagebox.showerror("ASCOM Unavailable", "Flask is not installed. Install flask and flask-cors to use the ASCOM Alpaca safety monitor.")
            return

        if self.ascom_server:
            return

        try:
            port = int(self.ascom_port.get())
            device_number = int(self.ascom_device_number.get())

            # Ensure a stable UniqueID exists and is persisted so NINA can reconnect
            # to the same device across reboots/restarts.
            if not self.ascom_unique_id:
                import uuid
                self.ascom_unique_id = str(uuid.uuid4())
                self.save_settings()

            self.ascom_server = AscomAlpacaSafetyMonitor(
                port=port,
                device_number=device_number,
                roof_classifier_app=self,
                unique_id=self.ascom_unique_id
            )
            
            # Start server in a separate thread
            server_thread = threading.Thread(
                target=self.ascom_server.run,
                daemon=True
            )
            server_thread.start()
            
            self.ascom_enabled.set(True)
            if self.logger:
                self.logger.info(f"ASCOM Alpaca server started on port {port}")

        except ValueError:
            if not silent:
                messagebox.showerror("Invalid ASCOM Settings", "The port and device number must be whole numbers.")
            elif self.logger:
                self.logger.error(
                    f"ASCOM auto-start skipped: invalid port {self.ascom_port.get()!r} "
                    f"or device number {self.ascom_device_number.get()!r}")
        except Exception as e:
            if not silent:
                messagebox.showerror("ASCOM Server Error", f"Could not start the ASCOM server: {e}")
            elif self.logger:
                self.logger.error(f"Failed to auto-start ASCOM server: {e}")

    def stop_ascom_server(self):
        """Stop the ASCOM Alpaca server"""
        if not self.ascom_server:
            return
        try:
            self.ascom_server.stop()
            self.ascom_server = None
            self.ascom_enabled.set(False)
            if self.logger:
                self.logger.info("ASCOM Alpaca server stopped")
        except Exception as e:
            messagebox.showerror("ASCOM Server Error", f"Could not stop the ASCOM server: {e}")

    def test_ascom_discovery(self):
        """Test ASCOM discovery functionality"""
        try:
            import socket
            import json
            
            # Send discovery packet
            discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            discovery_socket.settimeout(3.0)  # 3 second timeout
            
            # ASCOM discovery packet
            discovery_packet = b"alpacadiscovery1"
            discovery_socket.sendto(discovery_packet, ('127.0.0.1', 32227))
            
            try:
                # Wait for response
                data, addr = discovery_socket.recvfrom(1024)
                response = json.loads(data.decode('utf-8'))
                
                messagebox.showinfo("Discovery Test Success", 
                    f"Discovery response received from {addr}:\n\n"
                    f"Server: {response.get('ServerName', 'Unknown')}\n"
                    f"Port: {response.get('AlpacaPort', 'Unknown')}\n"
                    f"Manufacturer: {response.get('Manufacturer', 'Unknown')}\n"
                    f"Version: {response.get('ManufacturerVersion', 'Unknown')}")
                    
            except socket.timeout:
                messagebox.showwarning("Discovery Test Failed", 
                    "No discovery response received within 3 seconds.\n\n"
                    "Possible issues:\n"
                    "• ASCOM server not running\n"
                    "• Firewall blocking UDP port 32227\n"
                    "• Discovery not enabled")
                    
            finally:
                discovery_socket.close()
                
        except ImportError:
            messagebox.showerror("Error", "Socket module not available for discovery test.")
        except Exception as e:
            messagebox.showerror("Discovery Test Error", f"Error testing discovery: {str(e)}")
            
    def open_ascom_setup_page(self):
        """Open the ASCOM setup page in the default web browser"""
        try:
            import webbrowser
            port = self.ascom_port.get()
            url = f"http://localhost:{port}/setup"
            webbrowser.open(url)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open setup page: {str(e)}\n\n"
                                f"Try manually opening: http://localhost:{self.ascom_port.get()}/setup")

    def calculate_sun_angle(self, config=None):
        """Calculate the sun's elevation angle for the given location using UTC.

        *config* is an optional snapshot from _get_monitor_config; one is obtained
        automatically when omitted, which keeps worker-thread callers off Tk.
        """
        try:
            if config is None:
                config = self._get_monitor_config() or {}
            # Create observer for the given location
            observer = ephem.Observer()
            observer.lat = str(float(config['latitude']))
            observer.lon = str(float(config['longitude']))
            # Use UTC time for calculations
            observer.date = ephem.now()
            
            # Calculate sun position
            sun = ephem.Sun()
            sun.compute(observer)
            
            # Return elevation angle in degrees
            return float(sun.alt) * 180.0 / ephem.pi
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error calculating sun angle: {e}")
            # None means "unknown". It used to be 0.0, which is indistinguishable
            # from a real sunrise-altitude reading; callers must handle the
            # unknown case explicitly instead of acting on a fabricated angle.
            return None

    def is_sun_safe_for_open(self, config=None):
        """Check if sun angle is safe to report 'open' status.

        *config* is an optional snapshot from _get_monitor_config (see above).
        """
        try:
            if config is None:
                config = self._get_monitor_config() or {}
            sun_angle = self.calculate_sun_angle(config)
            if sun_angle is None:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.error(
                        "Sun altitude unknown - treating conditions as unsafe for OPEN")
                return False
            threshold = float(config['sun_angle_threshold'])
            is_safe = sun_angle < threshold

            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Sun angle: {sun_angle:.1f}°, Threshold: {threshold:.1f}°, Safe for open: {is_safe}")

            return is_safe
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error checking sun safety: {e}")
            # Fail closed. Reporting "safe to be open" on the strength of a failed
            # calculation is the one outcome that can expose equipment to daylight.
            return False

    # ── Thread-safe configuration snapshots ───────────────────────────────────
    #
    # Tk variables may only be touched from the thread running the main loop, but
    # the monitor loop and the ASCOM server both need the current configuration.
    # They therefore work from a plain-dict snapshot taken on the UI thread.

    def _snapshot_monitor_config(self):
        """Read every Tk variable the monitoring path needs into a plain dict.

        Must be called on the UI thread.
        """
        return {
            'camera_url': self.camera_url.get().strip(),
            'monitor_path': self.monitor_path.get(),
            'output_path': self.output_path.get(),
            'secondary_enabled': self.secondary_source_enabled.get(),
            'secondary_source': self.secondary_source_path.get().strip(),
            'latitude': self.latitude.get(),
            'longitude': self.longitude.get(),
            'sun_angle_threshold': self.sun_angle_threshold.get(),
            'sample_mode_enabled': self.sample_mode_enabled.get(),
            'sample_rate': self.sample_rate.get(),
            'training_data_folder': self.training_data_folder.get().strip(),
            'save_on_toggle': self.save_on_toggle_enabled.get(),
            'save_on_disagreement': self.save_on_disagreement_enabled.get(),
            # Notification settings: _check_and_send_notifications runs on the
            # monitor thread, so it must read these from the snapshot too.
            'notif_stale_enabled': self.notif_stale_enabled.get(),
            'notif_stale_minutes': self.notif_stale_minutes.get(),
            'notif_stale_url': self.notif_stale_url.get().strip(),
            'stale_image_action': self.stale_image_action.get(),
            'notif_open_enabled': self.notif_open_enabled.get(),
            'notif_open_url': self.notif_open_url.get().strip(),
            'notif_closed_enabled': self.notif_closed_enabled.get(),
            'notif_closed_url': self.notif_closed_url.get().strip(),
            'notif_heartbeat_enabled': self.notif_heartbeat_enabled.get(),
            'notif_heartbeat_minutes': self.notif_heartbeat_minutes.get(),
            'notif_heartbeat_url': self.notif_heartbeat_url.get().strip(),
        }

    def _request_monitor_config(self, timeout=5.0):
        """Ask the UI thread for a configuration snapshot and wait for it.

        Returns the most recent snapshot if the UI thread does not answer in time,
        or None if there has never been one.
        """
        result = {}
        done = threading.Event()

        def grab():
            try:
                result.update(self._snapshot_monitor_config())
            except Exception:
                pass
            finally:
                done.set()

        try:
            self.root.after(0, grab)
        except Exception:
            return self._last_monitor_config

        if done.wait(timeout) and result:
            self._last_monitor_config = result
            return result

        if self.logger:
            self.logger.warning("Timed out waiting for a configuration snapshot from the UI thread")
        return self._last_monitor_config

    def _get_monitor_config(self):
        """Return a configuration snapshot, however the caller's thread allows."""
        if threading.current_thread() is threading.main_thread():
            config = self._snapshot_monitor_config()
            self._last_monitor_config = config
            return config
        return self._request_monitor_config()

    @staticmethod
    def _is_http_source(path):
        """True if *path* points at an HTTP(S) resource rather than a local file."""
        return path.strip().lower().startswith(("http://", "https://"))

    @staticmethod
    def _parse_secondary_status(text):
        """Parse OPEN/CLOSED out of the last non-empty line of *text*.

        Returns (status, last_line) where status is None if nothing could be parsed.
        """
        return parse_roof_status(text)

    def _fetch_secondary_from_url(self, url):
        """Fetch and parse the secondary roof status from an HTTP(S) URL.

        Returns (status, mod_time) where mod_time comes from the Last-Modified header
        when the server provides one, otherwise the time of the fetch. Raises on
        network errors so the caller can log them.
        """
        req = urllib.request.Request(url, headers={"User-Agent": "SyntheticRoofStatus/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            raw = response.read()
            last_modified = response.headers.get("Last-Modified")

        text = raw.decode("utf-8", errors="replace")
        status, last_line = self._parse_secondary_status(text)
        if status is None:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Could not parse status from secondary source URL: {last_line}")
            return None, None

        mod_time = None
        if last_modified:
            try:
                from email.utils import parsedate_to_datetime
                parsed = parsedate_to_datetime(last_modified)
                # Normalise to a naive UTC datetime to match the local-file branch
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(pytz.utc).replace(tzinfo=None)
                mod_time = parsed
            except Exception:
                mod_time = None
        if mod_time is None:
            mod_time = datetime.utcnow()

        return status, mod_time

    def read_secondary_source(self, config=None):
        """Read the secondary source roof status from a local file or an HTTP(S) URL.

        Returns (status, mod_time_utc), or (None, None) when unavailable/unparseable.
        Pass *config* (a snapshot from _get_monitor_config) when calling from a worker
        thread; without it the Tk variables are read directly, which is only safe on
        the UI thread.
        """
        if config is None:
            config = self._get_monitor_config()
        if config is None:
            return None, None
        return self._read_secondary_values(config.get('secondary_enabled'),
                                           config.get('secondary_source', ''))

    def _read_secondary_values(self, enabled, source):
        """Secondary-source reader working purely from plain values — no Tk access,
        so it is safe on any thread.

        URL results are cached briefly so repeated UI refreshes do not issue a network
        request per call.
        """
        if not enabled or not source:
            return None, None

        source = source.strip()

        if self._is_http_source(source):
            # Serve from cache when it is still fresh and for the same URL
            if self._secondary_cache:
                cached_source, cached_status, cached_time, fetched_at = self._secondary_cache
                if (cached_source == source
                        and (datetime.utcnow() - fetched_at).total_seconds() < _SECONDARY_URL_CACHE_SECONDS):
                    return cached_status, cached_time
            try:
                status, mod_time = self._fetch_secondary_from_url(source)
                self._secondary_cache = (source, status, mod_time, datetime.utcnow())
                if status and hasattr(self, 'logger') and self.logger:
                    self.logger.info(f"Secondary source status: {status}, Last updated: {mod_time}")
                return status, mod_time
            except Exception as e:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.error(f"Error reading secondary source URL {source}: {e}")
                # Cache the failure too, so a dead URL does not stall every refresh
                self._secondary_cache = (source, None, None, datetime.utcnow())
                return None, None

        try:
            file_path = source
            if not os.path.exists(file_path):
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning(f"Secondary source file not found: {file_path}")
                return None, None

            # Get file modification time in UTC
            mod_time = datetime.utcfromtimestamp(os.path.getmtime(file_path))

            # Read the last line of the file
            with open(file_path, 'r') as f:
                content = f.read()

            if not content.strip():
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning(f"Secondary source file is empty: {file_path}")
                return None, None

            status, last_line = self._parse_secondary_status(content)
            if status is None:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning(f"Could not parse status from secondary source: {last_line}")
                return None, None

            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Secondary source status: {status}, Last updated: {mod_time}")

            return status, mod_time

        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error reading secondary source: {e}")
            return None, None

    # ── Look and feel ─────────────────────────────────────────────────────────

    def _px(self, value):
        """Scale a 96-DPI pixel size to the screen the window is on."""
        return int(round(value * getattr(self, "_ui_scale", 1.0)))

    def _init_style(self):
        """Fonts, spacing and ttk styles shared by the main window and dialogs."""
        self._ui_scale = max(1.0, self.root.winfo_fpixels("1i") / 96.0)
        self._preview_max = (self._px(_PREVIEW_IMG_MAX_W), self._px(_PREVIEW_IMG_MAX_H))
        self._left_column_wrap = self._px(400)

        base = tkfont.nametofont("TkDefaultFont")
        family, size = base.actual("family"), base.actual("size")
        self.font_small = tkfont.Font(family=family, size=max(size - 1, 8))
        self.font_bold = tkfont.Font(family=family, size=size, weight="bold")
        self.font_status = tkfont.Font(family=family, size=size + 13, weight="bold")

        style = ttk.Style(self.root)
        style.configure("TLabelframe.Label", font=self.font_bold, foreground=COLOR_TEXT)
        style.configure("Muted.TLabel", foreground=COLOR_MUTED, font=self.font_small)
        style.configure("Heading.TLabel", font=self.font_bold)
        style.configure("Treeview", rowheight=self._px(22))
        style.configure("TNotebook.Tab", padding=(self._px(10), self._px(3)))

    def _section(self, parent, title):
        """A titled group of related settings."""
        pad = self._px(10)
        return ttk.LabelFrame(parent, text=title, padding=(pad, self._px(6), pad, pad))

    def _hint(self, parent, text, **pack):
        """Secondary explanatory text that wraps to the width it is given."""
        label = ttk.Label(parent, text=text, style="Muted.TLabel", justify=tk.LEFT,
                          wraplength=self._px(340))
        label.bind("<Configure>",
                   lambda e: label.configure(wraplength=max(e.width - 2, self._px(120))))
        pack.setdefault("fill", "x")
        label.pack(**pack)
        return label

    def _status_text(self, parent, text="", fg=COLOR_MUTED, **options):
        """A plain Tk label for text whose colour carries meaning (ttk labels are
        styled per class, not per widget)."""
        options.setdefault("wraplength", self._left_column_wrap)
        return tk.Label(parent, text=text, fg=fg, anchor="w", justify=tk.LEFT, **options)

    def _field_row(self, parent, label, var, buttons=()):
        """A label above an entry, with optional buttons beside the entry.

        *buttons* is a sequence of ``(text, command)``. Returns ``(entry, [button, ...])``.
        """
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(0, self._px(8)))
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, columnspan=len(buttons) + 1,
                                          sticky="w", pady=(0, self._px(2)))
        entry = ttk.Entry(frame, textvariable=var, width=36)
        entry.grid(row=1, column=0, sticky="ew")
        made = []
        for column, (text, command) in enumerate(buttons, start=1):
            button = ttk.Button(frame, text=text, command=command)
            button.grid(row=1, column=column, padx=(self._px(6), 0))
            made.append(button)
        return entry, made

    def _button_row(self, parent, buttons, **pack):
        """Pack ``(text, command)`` buttons left to right; returns the buttons."""
        frame = ttk.Frame(parent)
        pack.setdefault("fill", "x")
        pack.setdefault("pady", (0, self._px(8)))
        frame.pack(**pack)
        made = []
        for text, command in buttons:
            button = ttk.Button(frame, text=text, command=command)
            button.pack(side=tk.LEFT, padx=(0, self._px(6)))
            made.append(button)
        return made

    @staticmethod
    def _set_enabled(widget, enabled):
        if isinstance(widget, ttk.Widget):
            widget.state(["!disabled"] if enabled else ["disabled"])
        else:
            widget.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _enable_with(self, var, *widgets):
        """Enable *widgets* only while the checkbox variable *var* is set."""
        def sync(*_):
            for widget in widgets:
                self._set_enabled(widget, bool(var.get()))
        var.trace_add("write", sync)
        sync()

    def _make_dialog(self, title, min_size=None):
        """A secondary window placed over the main one; Escape closes it."""
        win = tk.Toplevel(self.root)
        win.title(title)
        win.transient(self.root)
        win.bind("<Escape>", lambda e: win.destroy())
        if min_size:
            win.minsize(self._px(min_size[0]), self._px(min_size[1]))
        return win

    def _center_over_root(self, win, width=None, height=None):
        """Size *win* (to its content unless given) and centre it over the main window."""
        win.update_idletasks()
        width = max(width or 0, win.winfo_reqwidth())
        height = max(height or 0, win.winfo_reqheight())
        x = self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - height) // 3
        win.geometry(f"{width}x{height}+{max(x, 0)}+{max(y, 0)}")

    @staticmethod
    def _short_source(source, limit=48):
        """A file name or URL shortened for display."""
        if len(source) <= limit:
            return source
        keep = (limit - 1) // 2
        return f"{source[:keep]}…{source[-keep:]}"

    @staticmethod
    def _format_elapsed(seconds):
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds} s"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes} min"
        hours, minutes = divmod(minutes, 60)
        return f"{hours} h {minutes} min" if minutes else f"{hours} h"

    # ── Main window ───────────────────────────────────────────────────────────

    def setup_gui(self):
        self._init_style()
        self._save_after_id = None
        self._activity_after_id = None
        self._obs_refresh_after_id = None

        self._build_status_bar()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=self._px(8), pady=(self._px(8), self._px(4)))
        self._tabs = {}
        self._build_monitoring_tab()
        self._build_training_tab()
        self._build_configuration_tab()
        self._build_notifications_tab()
        self._build_utilities_tab()

        self._restore_window_state()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Every setting is saved shortly after it changes. Previously only buttons
        # and checkboxes saved, so anything typed into a field was lost on exit.
        for var in vars(self).values():
            if isinstance(var, tk.Variable):
                var.trace_add("write", self._schedule_save)
        for var in (self.latitude, self.longitude, self.sun_angle_threshold):
            var.trace_add("write", self._schedule_observation_refresh)

        self.update_training_stats()
        self._update_model_display()
        self._update_override_display()
        self._update_ascom_display()
        self._render_roof_status()
        self.root.after(1000, self._tick_override_display)
        self.root.after(300, self.update_observation_window_display)
        if self.preview_enabled.get():
            self.root.after(1200, self.refresh_preview)

    def _build_status_bar(self):
        bar = ttk.Frame(self.root, padding=(self._px(8), self._px(3), self._px(6), self._px(4)))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Separator(self.root, orient="horizontal").pack(side=tk.BOTTOM, fill=tk.X)

        self.statusbar_toggle_btn = ttk.Button(bar, text="Start Monitoring",
                                               command=self.toggle_monitoring)
        self.statusbar_toggle_btn.pack(side=tk.RIGHT)

        self.statusbar_label = tk.Label(bar, text="● Monitoring off", fg=COLOR_MUTED, anchor="w")
        self.statusbar_label.pack(side=tk.LEFT)
        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill="y", padx=self._px(10))
        self.statusbar_ascom_label = tk.Label(bar, text="", fg=COLOR_MUTED, anchor="w")
        self.statusbar_ascom_label.pack(side=tk.LEFT)
        ttk.Separator(bar, orient="vertical").pack(side=tk.LEFT, fill="y", padx=self._px(10))
        self.statusbar_model_label = tk.Label(bar, text="", fg=COLOR_MUTED, anchor="w")
        self.statusbar_model_label.pack(side=tk.LEFT)
        self.activity_label = tk.Label(bar, text="", fg=COLOR_MUTED, anchor="e")
        self.activity_label.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(self._px(10), self._px(10)))

    def _new_tab(self, key, title, columns=2):
        tab = ttk.Frame(self.notebook, padding=self._px(12))
        self.notebook.add(tab, text=title)
        self._tabs[key] = tab
        tab.rowconfigure(0, weight=1)
        cols = []
        for index in range(columns):
            tab.columnconfigure(index, weight=1, uniform="tabcols" if columns > 1 else None)
            col = ttk.Frame(tab)
            col.grid(row=0, column=index, sticky="nsew",
                     padx=(0, self._px(12) if index < columns - 1 else 0))
            cols.append(col)
        return tab, cols

    def _build_monitoring_tab(self):
        gap = self._px(12)
        tab = ttk.Frame(self.notebook, padding=gap)
        self.notebook.add(tab, text="Monitoring")
        self._tabs["monitor"] = tab
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)
        left = ttk.Frame(tab)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, gap))
        right = ttk.Frame(tab)
        right.grid(row=0, column=1, sticky="nsew")

        # Roof status: what is being reported right now, and why
        # (Start/Stop lives in the status bar, where it is reachable from every tab.)
        card = self._section(left, "Roof Status")
        card.pack(fill="x")
        self.roof_status_label = tk.Label(card, text="—", font=self.font_status,
                                          fg=COLOR_MUTED, anchor="w")
        self.roof_status_label.pack(fill="x")
        self.roof_status_note = self._status_text(card)
        self.roof_status_note.pack(fill="x")

        ttk.Separator(card).pack(fill="x", pady=self._px(8))
        check_row = ttk.Frame(card)
        check_row.pack(fill="x")
        self.countdown_label = self._status_text(check_row)
        self.countdown_label.pack(side=tk.RIGHT, anchor="n")
        self.status_label = self._status_text(check_row, "Not monitoring",
                                              wraplength=self._px(290))
        self.status_label.pack(side=tk.LEFT, fill="x", expand=True)
        self.hash_status_label = self._status_text(card, "Image: —")
        self.hash_status_label.pack(fill="x")
        self.sun_status_label = self._status_text(card, "Sun altitude: —")
        self.sun_status_label.pack(fill="x")
        # Packed only while a secondary source is configured (see _set_secondary_line)
        self.secondary_status_label = self._status_text(card)

        self._autostart_check = ttk.Checkbutton(card, text="Start monitoring when the app opens",
                                                variable=self.auto_start_monitoring)
        self._autostart_check.pack(anchor="w", pady=(self._px(8), 0))

        # Manual override
        override = self._section(left, "Manual Override")
        override.pack(fill="x", pady=(gap, 0))
        radios = ttk.Frame(override)
        radios.pack(fill="x", pady=(0, self._px(6)))
        for text, value in (("Use the model", "AUTO"), ("Force OPEN", "OPEN"),
                            ("Force CLOSED", "CLOSED")):
            ttk.Radiobutton(radios, text=text, variable=self.override_mode,
                            value=value).pack(side=tk.LEFT, padx=(0, self._px(14)))

        duration = ttk.Frame(override)
        duration.pack(fill="x", pady=(0, self._px(6)))
        ttk.Label(duration, text="For").pack(side=tk.LEFT)
        ttk.Combobox(duration, textvariable=self.override_duration, values=OVERRIDE_DURATIONS,
                     state="readonly", width=14).pack(side=tk.LEFT, padx=(self._px(6), self._px(10)))
        ttk.Button(duration, text="Apply",
                   command=self.apply_manual_override).pack(side=tk.LEFT, padx=(0, self._px(6)))
        self.override_clear_btn = ttk.Button(duration, text="Clear",
                                             command=self.clear_manual_override)
        self.override_clear_btn.pack(side=tk.LEFT)

        self.override_status_label = self._status_text(override)
        self.override_status_label.pack(fill="x", pady=(0, self._px(4)))
        self._hint(override,
                   "Forces the status written to the file and reported to ASCOM clients. "
                   "ASCOM still reports unsafe while the sun is above the threshold, "
                   "even under a forced OPEN.")

        # Latest image
        preview = self._section(right, "Latest Image")
        preview.pack(fill="both", expand=True)
        self.preview_bottom_frame = ttk.Frame(preview)
        self.preview_bottom_frame.pack(side=tk.BOTTOM, fill="x", pady=(self._px(6), 0))
        self.preview_refresh_btn = ttk.Button(self.preview_bottom_frame, text="Refresh",
                                              command=self.refresh_preview)
        self.preview_refresh_btn.pack(side=tk.RIGHT)
        ttk.Checkbutton(self.preview_bottom_frame, text="Show preview",
                        variable=self.preview_enabled,
                        command=self.on_preview_enabled_changed).pack(side=tk.RIGHT, padx=self._px(10))
        self.preview_caption_label = tk.Label(self.preview_bottom_frame, text="", fg=COLOR_MUTED,
                                              anchor="w", font=self.font_small)
        self.preview_caption_label.pack(side=tk.LEFT, fill="x", expand=True)

        width, height = self._preview_max
        self.preview_holder = tk.Frame(preview, bg=COLOR_VIEWER_BG, width=width, height=height)
        self.preview_holder.pack(side=tk.TOP, fill="both", expand=True)
        self.preview_holder.pack_propagate(False)
        self.preview_label = tk.Label(self.preview_holder, text="No image yet",
                                      fg=COLOR_VIEWER_TEXT, bg=COLOR_VIEWER_BG)
        self.preview_label.pack(fill="both", expand=True)

        if not self.preview_enabled.get():
            self.preview_holder.pack_forget()
            self.preview_refresh_btn.state(["disabled"])
            self.preview_caption_label.config(text="Preview hidden")

    def _build_training_tab(self):
        _tab, (left, right) = self._new_tab("training", "Training & Model")

        data = self._section(left, "Training Data")
        data.pack(fill="x")
        self._field_row(data, "Training data folder", self.training_data_folder,
                        [("Browse…", self.browse_training_data_folder)])
        self.training_data_folder.trace_add("write", lambda *_: self.update_training_stats())
        self.stats_label = ttk.Label(data, text="")
        self.stats_label.pack(anchor="w", pady=(0, self._px(8)))
        self._button_row(data, [("Add Open Images…", lambda: self.add_frame("open")),
                                ("Add Closed Images…", lambda: self.add_frame("closed"))])
        self._button_row(data, [("Review Unclassified…", self.open_classify_images_window),
                                ("Clear Training Data…", self.clear_training_data)])
        self._hint(data, "Images live in open/, closed/ and unclassified/ subfolders of this "
                         "folder. Leave it empty to use the folder the app runs from.")

        collect = self._section(left, "Collect Frames While Monitoring")
        collect.pack(fill="x", pady=(self._px(12), 0))
        sample = ttk.Frame(collect)
        sample.pack(fill="x", pady=(0, self._px(2)))
        ttk.Checkbutton(sample, text="Save a random sample of frames, at a rate of",
                        variable=self.sample_mode_enabled).pack(side=tk.LEFT)
        rate_entry = ttk.Entry(sample, textvariable=self.sample_rate, width=6)
        rate_entry.pack(side=tk.LEFT, padx=(self._px(4), 0))
        self._enable_with(self.sample_mode_enabled, rate_entry)
        ttk.Checkbutton(collect, text="Save the frame whenever the model's status changes",
                        variable=self.save_on_toggle_enabled).pack(anchor="w", pady=(0, self._px(2)))
        ttk.Checkbutton(collect, text="Save the first frame where the model disagrees with "
                                      "the secondary roof status",
                        variable=self.save_on_disagreement_enabled).pack(anchor="w", pady=(0, self._px(6)))
        self._hint(collect, "Collected frames go to unclassified/ so you can label them with "
                            "Review Unclassified. A rate of 0.1 keeps about one frame in ten.")

        model = self._section(right, "Model")
        model.pack(fill="x")
        self.model_status_label = self._status_text(model, font=self.font_bold)
        self.model_status_label.pack(fill="x")
        self.model_path_label = self._status_text(model, font=self.font_small)
        self.model_path_label.pack(fill="x", pady=(0, self._px(8)))
        _train, _load, save_as = self._button_row(model, [("Train Model", self.train_model),
                                                          ("Load Model…", self.load_model),
                                                          ("Save Model As…", self.save_current_model_as)])
        self._hint(model, "Training uses every image in open/ and closed/. If a model file is "
                          "loaded, training saves over it.")

        validation = self._section(right, "Validation")
        validation.pack(fill="x", pady=(self._px(12), 0))
        self._field_row(validation, "Fixed validation set", self.validation_set_path,
                        [("Browse…", self.browse_validation_set)])
        validate, _benchmark = self._button_row(validation, [("Validate Model", self.validate_model),
                                                             ("Benchmark Models…", self.benchmark_models)])
        # Enabled by _update_model_display once a model is loaded
        self._model_required_buttons = (save_as, validate)
        self._hint(validation, "A folder with open/ and closed/ subfolders, kept separate from "
                               "the training images. If it is empty, Validate asks for a folder.")

    def _build_configuration_tab(self):
        _tab, (left, right) = self._new_tab("config", "Configuration")
        gap = self._px(12)

        source = self._section(left, "Image Source")
        source.pack(fill="x")
        self._field_row(source, "Image folder", self.monitor_path,
                        [("Browse…", self.browse_monitor_folder)])
        self._field_row(source, "Camera image URL (optional)", self.camera_url,
                        [("Test", self._test_camera_url)])
        self._hint(source, "The newest image in the folder is checked every minute. If a camera "
                           "URL is set, the image is downloaded from it instead.")

        output = self._section(left, "Status Output")
        output.pack(fill="x", pady=(gap, 0))
        self._field_row(output, "Roof status file", self.output_path,
                        [("Browse…", self.browse_output_file)])
        self._hint(output, "Rewritten after every check with a single OPEN or CLOSED line, in "
                           "the roof file format that ASCOM safety monitor drivers read.")

        site = self._section(left, "Observatory")
        site.pack(fill="x", pady=(gap, 0))
        grid = ttk.Frame(site)
        grid.pack(fill="x", pady=(0, self._px(6)))
        ttk.Label(grid, text="Latitude").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.latitude, width=10).grid(
            row=0, column=1, sticky="w", padx=(self._px(6), self._px(18)))
        ttk.Label(grid, text="Longitude").grid(row=0, column=2, sticky="w")
        ttk.Entry(grid, textvariable=self.longitude, width=10).grid(
            row=0, column=3, sticky="w", padx=(self._px(6), 0))
        ttk.Label(grid, text="Sun limit (°)").grid(row=1, column=0, sticky="w", pady=(self._px(6), 0))
        ttk.Entry(grid, textvariable=self.sun_angle_threshold, width=10).grid(
            row=1, column=1, sticky="w", padx=(self._px(6), 0), pady=(self._px(6), 0))

        presets = ttk.Frame(site)
        presets.pack(fill="x", pady=(0, self._px(6)))
        for name, angle in TWILIGHT_PRESETS.items():
            label = "Horizon" if angle == "0.0" else name
            ttk.Button(presets, text=f"{label} ({float(angle):g}°)",
                       command=lambda p=name: self.apply_twilight_preset(p)).pack(
                side=tk.LEFT, padx=(0, self._px(4)))
        self._hint(site, "OPEN is only reported while the sun is below this altitude. "
                         "Longitude is positive east of Greenwich.")
        self.obs_window_label = self._status_text(site, "", fg=COLOR_TEXT)
        self.obs_window_label.pack(fill="x", pady=(self._px(6), 0))

        secondary = self._section(right, "Secondary Roof Status")
        secondary.pack(fill="x")
        ttk.Checkbutton(secondary, text="Compare with another roof status file or URL",
                        variable=self.secondary_source_enabled).pack(anchor="w", pady=(0, self._px(6)))
        entry, buttons = self._field_row(secondary, "File path or http(s) URL",
                                         self.secondary_source_path,
                                         [("Browse…", self.browse_secondary_source),
                                          ("Test", self._test_secondary_source)])
        self._enable_with(self.secondary_source_enabled, entry, *buttons)
        self._hint(secondary, "The last line must contain OPEN or CLOSED. Used for comparison "
                              "and logging only; it never changes the reported status.")

        ascom = self._section(right, "ASCOM Alpaca Safety Monitor")
        ascom.pack(fill="x", pady=(gap, 0))
        if FLASK_AVAILABLE:
            self.ascom_status_label = self._status_text(ascom, "", wraplength=self._px(360))
            self.ascom_status_label.pack(fill="x", pady=(0, self._px(8)))
            ports = ttk.Frame(ascom)
            ports.pack(fill="x", pady=(0, self._px(8)))
            ttk.Label(ports, text="Port").pack(side=tk.LEFT)
            self.ascom_port_entry = ttk.Entry(ports, textvariable=self.ascom_port, width=7)
            self.ascom_port_entry.pack(side=tk.LEFT, padx=(self._px(6), self._px(18)))
            ttk.Label(ports, text="Device number").pack(side=tk.LEFT)
            self.ascom_device_entry = ttk.Entry(ports, textvariable=self.ascom_device_number, width=5)
            self.ascom_device_entry.pack(side=tk.LEFT, padx=(self._px(6), 0))
            self.ascom_toggle_btn, self.ascom_discovery_btn, self.ascom_setup_btn = self._button_row(
                ascom, [("Start Server", self.toggle_ascom_server),
                        ("Test Discovery", self.test_ascom_discovery),
                        ("Open Setup Page", self.open_ascom_setup_page)])
            ttk.Checkbutton(ascom, text="Start the server when the app opens",
                            variable=self.auto_start_ascom).pack(anchor="w", pady=(0, self._px(6)))
            self._hint(ascom, "NINA and other Alpaca clients find this device automatically "
                              "(UDP discovery on port 32227). To add it by hand, use this "
                              "computer's address with the port and device number above.")
        else:
            self._hint(ascom, "Requires Flask and Flask-CORS. Install them with "
                              "pip install flask flask-cors, then restart the app.")

        logging_section = self._section(right, "Logging")
        logging_section.pack(fill="x", pady=(gap, 0))
        ttk.Checkbutton(logging_section, text="Write a log file",
                        variable=self.log_enabled,
                        command=self.on_log_enabled_changed).pack(anchor="w", pady=(0, self._px(6)))
        entry, buttons = self._field_row(logging_section, "Log file", self.log_path,
                                         [("Browse…", self.browse_log_file)])
        self._enable_with(self.log_enabled, entry, *buttons)

    def _build_notifications_tab(self):
        tab, (left, right) = self._new_tab("notifications", "Notifications")
        gap = self._px(12)

        def webhook(section, enabled_var, url_var, check_text, extra=None):
            ttk.Checkbutton(section, text=check_text,
                            variable=enabled_var).pack(anchor="w", pady=(0, self._px(6)))
            widgets = list(extra or [])
            entry, buttons = self._field_row(section, "Webhook URL", url_var,
                                             [("Test", lambda: self._test_webhook(url_var))])
            self._enable_with(enabled_var, entry, *buttons, *widgets)

        stale = self._section(left, "Stale Image")
        stale.pack(fill="x")
        threshold = ttk.Frame(stale)
        threshold.pack(fill="x", pady=(0, self._px(6)))
        ttk.Label(threshold, text="An image is stale after").pack(side=tk.LEFT)
        ttk.Entry(threshold, textvariable=self.notif_stale_minutes, width=5).pack(
            side=tk.LEFT, padx=self._px(6))
        ttk.Label(threshold, text="minutes without changing.").pack(side=tk.LEFT)
        ttk.Label(stale, text="While the image is stale, report:").pack(anchor="w")
        ttk.Radiobutton(stale, text="The model's classification of the last frame",
                        variable=self.stale_image_action,
                        value=STALE_ACTION_KEEP).pack(anchor="w", padx=(self._px(12), 0))
        ttk.Radiobutton(stale, text="CLOSED (fail-safe), since a frozen frame can't be trusted",
                        variable=self.stale_image_action,
                        value=STALE_ACTION_CLOSED).pack(anchor="w", padx=(self._px(12), 0))
        ttk.Separator(stale).pack(fill="x", pady=self._px(8))
        webhook(stale, self.notif_stale_enabled, self.notif_stale_url,
                "Send a webhook when the image goes stale")

        opened = self._section(left, "Roof Opened")
        opened.pack(fill="x", pady=(gap, 0))
        webhook(opened, self.notif_open_enabled, self.notif_open_url,
                "Send a webhook when the roof opens")

        heartbeat = self._section(right, "Heartbeat")
        heartbeat.pack(fill="x")
        ttk.Checkbutton(heartbeat, text="Send a heartbeat while monitoring",
                        variable=self.notif_heartbeat_enabled).pack(anchor="w", pady=(0, self._px(6)))
        interval = ttk.Frame(heartbeat)
        interval.pack(fill="x", pady=(0, self._px(8)))
        ttk.Label(interval, text="Every").pack(side=tk.LEFT)
        interval_entry = ttk.Entry(interval, textvariable=self.notif_heartbeat_minutes, width=5)
        interval_entry.pack(side=tk.LEFT, padx=self._px(6))
        ttk.Label(interval, text="minutes").pack(side=tk.LEFT)
        entry, buttons = self._field_row(heartbeat, "Webhook URL", self.notif_heartbeat_url,
                                         [("Test", lambda: self._test_webhook(self.notif_heartbeat_url))])
        self._enable_with(self.notif_heartbeat_enabled, interval_entry, entry, *buttons)
        self._hint(heartbeat, "Skipped while the image is stale, so a missing heartbeat means "
                              "monitoring or the camera needs attention.")

        closed = self._section(right, "Roof Closed")
        closed.pack(fill="x", pady=(gap, 0))
        webhook(closed, self.notif_closed_enabled, self.notif_closed_url,
                "Send a webhook when the roof closes")

        self._hint(right,"Webhooks are sent as an HTTP POST with a JSON body containing event, "
                         "status and timestamp, plus stale_minutes for stale-image events. "
                         "Open and closed events fire only when the status changes.",
                   pady=(gap, 0))

    def _build_utilities_tab(self):
        _tab, (left, _right) = self._new_tab("utilities", "Utilities")
        fits_section = self._section(left, "FITS to PNG")
        fits_section.pack(fill="x")
        self._hint(fits_section, "Convert FITS frames from an astronomy camera to PNG, for example "
                                 "to use them as training images. Debayering and stretching are "
                                 "optional.", pady=(0, self._px(8)))
        button = ttk.Button(fits_section, text="Convert FITS Files…", command=self.convert_fits_to_png)
        button.pack(anchor="w")
        if not FITS_AVAILABLE:
            button.state(["disabled"])
            self._hint(fits_section, "Requires astropy: pip install astropy",
                       pady=(self._px(6), 0))

    # ── Window state, autosave and small status helpers ───────────────────────

    def _select_tab(self, key):
        tab = getattr(self, "_tabs", {}).get(key)
        if tab is not None:
            self.notebook.select(tab)

    def _on_tab_changed(self, _event=None):
        self._last_tab = self.notebook.index(self.notebook.select())
        self._schedule_save()

    def _restore_window_state(self):
        """Size the window to its content, then reapply the saved size and tab."""
        self.root.update_idletasks()
        min_w, min_h = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
        self.root.minsize(min_w, min_h)

        match = re.fullmatch(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", getattr(self, "_window_geometry", "") or "")
        if match:
            w, h, x, y = (int(v) for v in match.groups())
            w, h = max(w, min_w), max(h, min_h)
            on_screen = (0 <= x < self.root.winfo_screenwidth() - 80
                         and 0 <= y < self.root.winfo_screenheight() - 80)
            self.root.geometry(f"{w}x{h}+{x}+{y}" if on_screen else f"{w}x{h}")
        if getattr(self, "_window_zoomed", False):
            try:
                self.root.state("zoomed")
            except tk.TclError:
                pass

        tab_index = getattr(self, "_last_tab", None)
        if not self.model and not self.model_path.get():
            # First run: there is nothing to monitor with until a model exists
            self._select_tab("training")
        elif isinstance(tab_index, int) and 0 <= tab_index < len(self.notebook.tabs()):
            self.notebook.select(tab_index)

    def _remember_window_state(self):
        try:
            self._window_zoomed = self.root.state() == "zoomed"
            if self.root.state() == "normal":
                self._window_geometry = self.root.geometry()
        except tk.TclError:
            pass

    def _schedule_save(self, *_):
        """Save settings once edits pause, rather than on every keystroke."""
        if self._save_after_id is not None:
            self.root.after_cancel(self._save_after_id)
        self._save_after_id = self.root.after(SETTINGS_SAVE_DELAY_MS, self._flush_settings)

    def _flush_settings(self):
        if getattr(self, "_save_after_id", None) is not None:
            self.root.after_cancel(self._save_after_id)
        self._save_after_id = None
        self.save_settings()

    def _schedule_observation_refresh(self, *_):
        if self._obs_refresh_after_id is not None:
            self.root.after_cancel(self._obs_refresh_after_id)
        self._obs_refresh_after_id = self.root.after(500, self._refresh_observation_now)

    def _refresh_observation_now(self):
        self._obs_refresh_after_id = None
        self.update_observation_window_display()

    def _set_activity(self, text, fg=COLOR_MUTED, clear_after_ms=10000):
        """Show a short-lived message at the bottom of the window."""
        if not hasattr(self, "activity_label"):
            return
        self.activity_label.config(text=text, fg=fg)
        if self._activity_after_id is not None:
            self.root.after_cancel(self._activity_after_id)
            self._activity_after_id = None
        if clear_after_ms:
            self._activity_after_id = self.root.after(
                clear_after_ms, lambda: self.activity_label.config(text=""))

    def _on_close(self):
        """Confirm before a close that would stop monitoring, then shut down cleanly."""
        if self.monitoring_active and not messagebox.askyesno(
                "Quit Synthetic RoofStatusFile",
                "Monitoring is running. If you quit, the roof status file stops "
                "updating and ASCOM clients lose the safety monitor.\n\nQuit anyway?",
                icon=messagebox.WARNING, default=messagebox.NO, parent=self.root):
            return
        self._remember_window_state()
        self._flush_settings()
        if self._monitor_stop_event is not None:
            self._monitor_stop_event.set()
        if self.ascom_server:
            try:
                self.ascom_server.stop()
            except Exception:
                pass
        if self._preview_tmp_path:
            try:
                os.unlink(self._preview_tmp_path)
            except OSError:
                pass
        self.root.destroy()

    def _update_model_display(self):
        """Show which model is loaded, on the Training tab and in the status bar."""
        if not hasattr(self, "model_status_label"):
            return
        for button in self._model_required_buttons:
            self._set_enabled(button, self.model is not None)
        path = self.model_path.get()
        if self.model is None:
            self.model_status_label.config(text="No model loaded", fg=COLOR_WARN)
            self.model_path_label.config(
                text="Train a model from your labelled images, or load a .joblib file.")
            self.statusbar_model_label.config(text="No model", fg=COLOR_WARN)
            return
        name = os.path.basename(path) if path else "Unsaved model"
        self.model_status_label.config(text=f"Loaded: {name}", fg=COLOR_OK)
        self.model_path_label.config(
            text=path or "Not saved yet. Use Save Model As… to keep it.")
        self.statusbar_model_label.config(text=f"Model: {self._short_source(name, 32)}",
                                          fg=COLOR_MUTED)

    def toggle_ascom_server(self):
        if self.ascom_server:
            self.stop_ascom_server()
        else:
            self.start_ascom_server()
        self._update_ascom_display()

    def _update_ascom_display(self):
        """Refresh the ASCOM state on the Configuration tab and in the status bar."""
        if not hasattr(self, "statusbar_ascom_label"):
            return
        server = self.ascom_server
        if not FLASK_AVAILABLE:
            self.statusbar_ascom_label.config(text="ASCOM unavailable", fg=COLOR_MUTED)
            return
        if server is None:
            bar, detail, fg = "ASCOM off", "Server stopped.", COLOR_MUTED
        elif not server.connected:
            bar = "ASCOM waiting for client"
            detail = f"Running on port {server.port}, device {server.device_number}. No client connected."
            fg = COLOR_TEXT
        else:
            safe, error = server.reported_safety()
            bar = f"ASCOM {'Safe' if safe else 'Unsafe'}"
            detail = (f"Running on port {server.port}, device {server.device_number}. "
                      f"Client connected; reporting {'Safe' if safe else 'Unsafe'}")
            detail += f" ({error})." if error and not safe else "."
            fg = COLOR_OK if safe else COLOR_WARN
        self.statusbar_ascom_label.config(text=bar, fg=fg if server else COLOR_MUTED)
        if hasattr(self, "ascom_status_label"):
            self.ascom_status_label.config(text=detail, fg=fg)
            running = server is not None
            self.ascom_toggle_btn.config(text="Stop Server" if running else "Start Server")
            self._set_enabled(self.ascom_port_entry, not running)
            self._set_enabled(self.ascom_device_entry, not running)
            self._set_enabled(self.ascom_discovery_btn, running)
            self._set_enabled(self.ascom_setup_btn, running)

    def _render_roof_status(self):
        """Refresh the large roof status on the Monitoring tab from the current state."""
        if not hasattr(self, "roof_status_label"):
            return
        snapshot = self._last_classification
        override = self.override_active
        if override:
            status, note, fg = override, "Manual override", COLOR_WARN
        elif self._failsafe_active:
            status = "CLOSED"
            note, fg = "Fail-safe: no valid classification for several minutes", COLOR_WARN
        elif snapshot:
            status = snapshot[1]
            reason = getattr(self, "_last_status_reason", "")
            note = reason or "Classified by the model"
            fg = COLOR_WARN if reason else COLOR_MUTED
            age = (datetime.now(timezone.utc) - snapshot[2]).total_seconds()
            if not self.monitoring_active:
                note += f". Monitoring is off; this result is {self._format_elapsed(age)} old"
            elif age > CLASSIFICATION_MAX_AGE_SECONDS:
                note += f". Result is {self._format_elapsed(age)} old"
                fg = COLOR_WARN
        else:
            status = None
            fg = COLOR_MUTED
            if self.model is None:
                note = "No model loaded. Train or load one on the Training & Model tab."
            elif self.monitoring_active:
                note = "Waiting for the first check…"
            else:
                note = "Not monitoring"
        self.roof_status_label.config(
            text=status or "—", fg=ROOF_STATUS_COLORS.get(status, COLOR_MUTED))
        self.roof_status_note.config(text=note, fg=fg)

    def _sync_monitoring_controls(self):
        button = getattr(self, "statusbar_toggle_btn", None)
        if button is not None:
            button.config(text="Stop Monitoring" if self.monitoring_active else "Start Monitoring")

    def _get_training_class_folder(self, label, base=None):
        """Return the full path to a training class subfolder (open/closed/unclassified/other).

        If a training_data_folder has been configured it is used as the base; otherwise the
        folder name is returned as-is for backward-compatible relative-path behaviour.
        Worker threads pass *base* from a configuration snapshot rather than letting this
        read the Tk variable.
        """
        base = self.training_data_folder.get().strip() if base is None else base.strip()
        if base:
            return os.path.join(base, label)
        return label

    def get_image_hash(self, image_path):
        """Generate a hash of the image content to detect duplicates"""
        return hashlib.md5(self.prep_image(image_path).tobytes()).hexdigest()

    def get_existing_hashes(self, folder_path):
        """Get hashes of all existing images in the given training folder"""
        hashes = set()
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        hash_val = self.get_image_hash(os.path.join(folder_path, file))
                        hashes.add(hash_val)
                    except Exception:
                        continue
        return hashes

    def add_frame(self, label):
        files = filedialog.askopenfilenames(filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")])
        if not files:
            return
        folder = self._get_training_class_folder(label)
        os.makedirs(folder, exist_ok=True)

        existing_hashes = self.get_existing_hashes(folder)
        count = 0
        duplicates = 0

        for path in files:
            # Check if this image is already in the training set
            try:
                img_hash = self.get_image_hash(path)
                if img_hash in existing_hashes:
                    duplicates += 1
                    continue

                dest = os.path.join(folder, os.path.basename(path))
                # If file exists, add a number suffix
                base, ext = os.path.splitext(dest)
                counter = 1
                while os.path.exists(dest):
                    dest = f"{base}_{counter}{ext}"
                    counter += 1
                with open(path, "rb") as fsrc, open(dest, "wb") as fdst:
                    fdst.write(fsrc.read())
                existing_hashes.add(img_hash)
                count += 1
            except Exception as e:
                print(f"Error processing {path}: {e}")

        message = f"Added {count} {label} image{'s' if count != 1 else ''}"
        if duplicates > 0:
            message += f"; skipped {duplicates} already in the training set"
        self._set_activity(message + ".")
        self.update_training_stats()

    def update_training_stats(self):
        """Update the training statistics display"""
        def _count(label):
            folder = self._get_training_class_folder(label)
            if os.path.isdir(folder):
                return len([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            return 0

        open_count = _count("open")
        closed_count = _count("closed")
        unclassified_count = _count("unclassified")

        text = f"{open_count} open  ·  {closed_count} closed"
        if unclassified_count:
            text += f"  ·  {unclassified_count} waiting for review"
        self.stats_label.config(text=text)

    def clear_training_data(self):
        """Clear all training data"""
        folders = [self._get_training_class_folder(label) for label in ("open", "closed")]
        if not any(os.path.isdir(folder) for folder in folders):
            messagebox.showinfo("Clear Training Data", "There is no training data to clear.")
            return
        if not messagebox.askyesno(
                "Clear Training Data",
                "Permanently delete every image in these folders?\n\n"
                + "\n".join(os.path.abspath(folder) for folder in folders)
                + "\n\nUnclassified images are kept. This cannot be undone.",
                icon=messagebox.WARNING, default=messagebox.NO):
            return

        for folder in folders:
            if os.path.isdir(folder):
                shutil.rmtree(folder)
        self._set_activity("Training data cleared.")
        self.update_training_stats()

    def _load_training_data(self):
        """Load the labelled training images as ``(X, y, skipped)``.

        An unreadable file (truncated, or not really an image) is listed in
        *skipped* instead of aborting the whole training run.
        """
        X, y, skipped = [], [], []
        for label, val in [("open", 1), ("closed", 0)]:
            folder = self._get_training_class_folder(label)
            if not os.path.isdir(folder):
                continue
            for file in os.listdir(folder):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        img = self.prep_image(os.path.join(folder, file))
                    except ValueError:
                        skipped.append(f"{label}/{file}")
                        continue
                    X.append(img.flatten())
                    y.append(val)
        return X, y, skipped

    def train_model(self):
        X, y, skipped = self._load_training_data()
        if not X:
            messagebox.showerror(
                "No Training Data",
                "There are no labelled images to train on. Add some with "
                "\"Add Open Images\" and \"Add Closed Images\".")
            return
        if len(set(y)) < 2:
            # LogisticRegression.fit raises on a single class, which used to
            # escape the button handler with no message at all.
            missing = "closed" if 1 in y else "open"
            messagebox.showerror(
                "Not Enough Training Data",
                f"Training needs both open and closed examples, but there are no "
                f"{missing} images. Add some with \"Add {missing.capitalize()} Images\".")
            return
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)

        # Show training summary
        open_count = sum(1 for label in y if label == 1)
        closed_count = sum(1 for label in y if label == 0)
        message = f"Model trained on {open_count} open and {closed_count} closed images."
        # Things the user needs to act on. A clean run is reported in the status
        # bar; only these warrant a dialog.
        notes = []
        if skipped:
            note = f"Skipped {len(skipped)} unreadable image(s): {', '.join(skipped[:5])}"
            notes.append(note + (", ..." if len(skipped) > 5 else ""))

        # Keep the trained model even if saving it fails - it is still usable.
        self.model = clf
        if self.model_path.get():
            try:
                dump(clf, self.model_path.get())
                message += f" Saved to {os.path.basename(self.model_path.get())}."
                self.save_settings()
            except Exception as e:
                notes.append(f"WARNING: could not save the model to {self.model_path.get()}: {e}")
        else:
            notes.append("The model has not been saved. Use Save Model As… to keep it.")
        self._update_model_display()
        self._render_roof_status()
        if notes:
            messagebox.showinfo("Model Trained", message + "\n\n" + "\n\n".join(notes))
        else:
            self._set_activity(message)

    @staticmethod
    def _load_model_file(path):
        """Load a .joblib model and check it can classify this app's frames.

        Raises ValueError with a readable message otherwise. A model trained at a
        different image size (or anything that is not a classifier) used to load
        without complaint and then fail every single monitoring pass all night.
        """
        model = load(path)
        if not callable(getattr(model, "predict", None)):
            raise ValueError(f"{os.path.basename(path)} is not a classifier model.")
        expected = IMG_SIZE * IMG_SIZE
        n_features = getattr(model, "n_features_in_", None)
        if n_features != expected:
            raise ValueError(
                f"{os.path.basename(path)} reports {n_features!r} input features, but this "
                f"app produces {expected} ({IMG_SIZE}x{IMG_SIZE} pixels). Retrain the model.")
        return model

    def load_model(self):
        current = self.model_path.get()
        path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(current)) if current else os.getcwd())
        if path:
            try:
                self.model = self._load_model_file(path)
            except Exception as e:
                messagebox.showerror("Could Not Load Model", f"Failed to load model: {e}")
                return
            self.model_path.set(path)
            self.save_settings()
            self._update_model_display()
            self._render_roof_status()
            self._set_activity(f"Loaded model {os.path.basename(path)}.")

    def prep_image(self, path):
        """Load *path* as the small grayscale array the model works on.

        Raises ValueError when the file cannot be decoded. cv2.imread reports
        that by returning None, which used to reach cv2.resize as an opaque
        cv2.error - on the monitor thread that ended monitoring for good whenever
        the camera was still writing the newest frame.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image {path}")
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    def validate_model(self):
        """Run validation on a set of test images.

        Uses the fixed validation set if one is configured; otherwise prompts the
        user to choose a folder.
        """
        if not self.model:
            messagebox.showerror("No Model Loaded", "Train or load a model before validating it.")
            return

        fixed = self.validation_set_path.get().strip()
        if fixed:
            folder = fixed
        else:
            folder = filedialog.askdirectory(title="Select folder with validation images")
            if not folder:
                return

        try:
            X_val, y_val, file_names = self._load_validation_data(folder)
        except ValueError as e:
            messagebox.showerror("Validation Set Error", str(e))
            return

        y_pred = self.model.predict(np.array(X_val))
        accuracy = accuracy_score(y_val, y_pred)
        self.show_validation_results(y_val, y_pred, file_names, accuracy, folder)

    def _scrolled_tree(self, parent, columns, scroll_x=False, scroll_y=True):
        """A Treeview with scrollbars, packed into *parent*. Returns the tree."""
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=[key for key, _title, _width in columns],
                            show="headings", selectmode="browse")
        for key, title, width in columns:
            tree.heading(key, text=title, anchor="w")
            tree.column(key, width=self._px(width), anchor="w", stretch=key == columns[0][0])
        tree.grid(row=0, column=0, sticky="nsew")
        if scroll_y:
            vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vertical.set)
            vertical.grid(row=0, column=1, sticky="ns")
        if scroll_x:
            horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
            tree.configure(xscrollcommand=horizontal.set)
            horizontal.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _open_image_on_double_click(self, tree, folder):
        """Open the double-clicked row's image (first column is its path under *folder*)."""
        if not folder or not hasattr(os, "startfile"):
            return

        def on_double_click(_event):
            selected = tree.focus()
            if not selected:
                return
            path = os.path.join(folder, tree.item(selected, "values")[0])
            try:
                os.startfile(path)
            except OSError as e:
                messagebox.showerror("Could Not Open Image", str(e), parent=tree)

        tree.bind("<Double-1>", on_double_click)

    def show_validation_results(self, y_true, y_pred, file_names, accuracy, folder=None):
        """Display validation results in a new window"""
        win = self._make_dialog("Validation Results", min_size=(520, 380))
        body = ttk.Frame(win, padding=self._px(14))
        body.pack(fill="both", expand=True)

        total = len(y_true)
        mistakes = sum(1 for t, p in zip(y_true, y_pred) if t != p)

        header = ttk.Frame(body)
        header.pack(fill="x", pady=(0, self._px(12)))
        tk.Label(header, text=f"{accuracy * 100:.1f}%", font=self.font_status,
                 fg=COLOR_OK if mistakes == 0 else COLOR_TEXT).pack(side=tk.LEFT, anchor="n")
        summary = ttk.Frame(header)
        summary.pack(side=tk.LEFT, padx=(self._px(12), 0), anchor="n", pady=(self._px(4), 0))
        ttk.Label(summary, text="Accuracy", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(summary, text=f"{total - mistakes} of {total} images classified correctly").pack(anchor="w")

        # Rows are the true class, columns the prediction. Fixed labels keep the
        # matrix 2x2 even when the validation set holds only one class.
        (tp, fn), (fp, tn) = confusion_matrix(y_true, y_pred, labels=[1, 0])
        matrix = ttk.Frame(header)
        matrix.pack(side=tk.RIGHT, anchor="n")
        cells = [("", "Predicted OPEN", "Predicted CLOSED"),
                 ("Actually OPEN", tp, fn),
                 ("Actually CLOSED", fp, tn)]
        for r, row in enumerate(cells):
            for c, value in enumerate(row):
                wrong = (r, c) in ((1, 2), (2, 1)) and value
                ttk.Label(matrix, text=str(value),
                          font=self.font_bold if wrong else None,
                          foreground=COLOR_ERROR if wrong else "",
                          style="Muted.TLabel" if r == 0 or c == 0 else "TLabel").grid(
                    row=r, column=c, sticky="e" if c else "w", padx=(self._px(10), 0), pady=1)

        controls = ttk.Frame(body)
        controls.pack(fill="x", pady=(0, self._px(6)))
        only_mistakes = tk.BooleanVar(value=mistakes > 0)
        ttk.Checkbutton(controls, text=f"Show only mistakes ({mistakes})",
                        variable=only_mistakes, command=lambda: fill()).pack(side=tk.LEFT)
        if folder and hasattr(os, "startfile"):
            ttk.Label(controls, text="Double-click an image to open it.",
                      style="Muted.TLabel").pack(side=tk.RIGHT)

        tree = self._scrolled_tree(body, [("file", "Image", 320), ("actual", "Actual", 90),
                                          ("predicted", "Predicted", 90), ("result", "Result", 80)])
        tree.tag_configure("wrong", foreground=COLOR_ERROR)
        self._open_image_on_double_click(tree, folder)

        def fill():
            tree.delete(*tree.get_children())
            for true_label, pred_label, filename in zip(y_true, y_pred, file_names):
                correct = true_label == pred_label
                if only_mistakes.get() and correct:
                    continue
                tree.insert("", tk.END, tags=() if correct else ("wrong",), values=(
                    filename, "OPEN" if true_label == 1 else "CLOSED",
                    "OPEN" if pred_label == 1 else "CLOSED",
                    "Correct" if correct else "Wrong"))

        fill()
        self._center_over_root(win, self._px(680), self._px(520))

    def classify_latest_png(self, config=None, max_cache_age=CLASSIFICATION_CACHE_SECONDS):
        """Classify the newest frame and write the roof status file.

        Runs on the monitor thread and on the ASCOM server thread as well as the UI
        thread. Only one classification happens at a time: concurrent callers would
        otherwise interleave writes to the status file and corrupt the shared
        toggle/disagreement state. A caller arriving while a result less than
        *max_cache_age* seconds old is available reuses that result instead of
        re-running the pipeline; pass ``max_cache_age=0`` to force a fresh pass.
        A fresh pass that fails clears the cached result.
        """
        with self._classify_lock:
            cached = self.get_cached_classification(max_cache_age)
            if cached is not None:
                filename, status, _age = cached
                return filename, status
            try:
                result = self._classify_latest_png_uncached(config)
            except BaseException:
                self._last_classification = None
                raise
            if result[0] is None:
                # A pass that could not classify must not leave the previous
                # result standing: the ASCOM thread would keep reporting it as
                # current until it aged out.
                self._last_classification = None
            return result

    def get_cached_classification(self, max_age_seconds):
        """Return ``(filename, status, age_seconds)`` if a recent enough result exists.

        Returns None when nothing has been classified yet or the last result is
        older than *max_age_seconds*.
        """
        if max_age_seconds <= 0:
            return None  # caller explicitly wants a fresh pass
        snapshot = self._last_classification
        if not snapshot:
            return None
        filename, status, taken_at = snapshot
        age = (datetime.now(timezone.utc) - taken_at).total_seconds()
        if age > max_age_seconds:
            return None
        return filename, status, age

    def get_cached_status(self, max_age_seconds=CLASSIFICATION_MAX_AGE_SECONDS):
        """Return ``(status, age_seconds)`` for the ASCOM safety monitor.

        ``status`` is None when no usable recent classification exists, in which
        case the caller must treat conditions as unsafe rather than reusing an old
        answer. ``age_seconds`` is None only when nothing has ever been classified.

        An active manual override is reported as the current status. The cached
        classification only picks an override up on the next monitoring pass, so
        a Force CLOSED used to leave IsSafe True for up to a minute and a half -
        and indefinitely while monitoring was stopped.
        """
        override = self.get_manual_override()
        if override:
            return override, 0.0
        snapshot = self._last_classification
        if not snapshot:
            return None, None
        _filename, status, taken_at = snapshot
        age = (datetime.now(timezone.utc) - taken_at).total_seconds()
        if age > max_age_seconds:
            return None, age
        return status, age

    def _classify_latest_png_uncached(self, config=None):
        """Run one full classification pass. Callers must hold _classify_lock."""
        if not self.model:
            if self.logger:
                self.logger.error("No model loaded")
            return None, "No model loaded"

        if config is None:
            config = self._get_monitor_config()
        if config is None:
            if self.logger:
                self.logger.error("No configuration snapshot available")
            return None, "No configuration available"

        img_path, latest, is_temp, error = self._resolve_latest_image(
            config['camera_url'], config['monitor_path'])
        if img_path is None:
            if self.logger:
                self.logger.error(f"Could not obtain an image to classify: {error}")
            return None, error

        try:
            # Decode before anything else touches the frame: a half-written file
            # must fail this pass cleanly, not get sampled into the training set.
            try:
                img = self.prep_image(img_path).flatten().reshape(1, -1)
            except ValueError as e:
                if self.logger:
                    self.logger.error(f"Skipping unreadable image {latest}: {e}")
                return None, f"Could not read image {latest}"
            return self._classify_image(img, img_path, latest, is_temp, config)
        finally:
            # A URL download is ours to delete, whatever happened above.
            if is_temp:
                try:
                    os.unlink(img_path)
                except OSError:
                    pass

    def _classify_image(self, img, img_path, latest, is_temp, config):
        """Classify a decoded frame and write the status file (see above)."""
        if not is_temp:
            # Optionally save a random sample for manual classification
            self.save_sample_if_needed(img_path, config)

        # Show the frame we are about to classify on the Monitoring tab
        self._capture_preview(img_path, latest)

        # Track image file hash to detect when a new image arrives
        try:
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash != self.last_image_hash:
                self.last_image_hash = file_hash
                self.last_new_hash_time = datetime.utcnow()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not compute image hash for {img_path}: {e}")

        # Get secondary source status for comparison
        secondary_status, secondary_time = self.read_secondary_source(config)
        # Kept for the Monitoring tab, which must not repeat this (possibly
        # network) read on the UI thread.
        self._last_secondary = (secondary_status, secondary_time)

        # Classify the image
        pred = self.model.predict(img)[0]
        image_status = "OPEN" if pred == 1 else "CLOSED"
        
        # Apply sun angle guard rails
        sun_safe = self.is_sun_safe_for_open(config)
        if image_status == "OPEN" and not sun_safe:
            if self.logger:
                self.logger.warning(f"Image classification suggests OPEN, but sun angle too high - overriding to CLOSED")
            final_status = "CLOSED"
            override_reason = " (Sun too high - safety override)"
        else:
            final_status = image_status
            override_reason = ""

        # The status the app would report with no manual override in force. Toggle
        # capture is baselined on this so that applying or clearing an override never
        # looks like a model transition.
        model_status = final_status

        # A frozen feed (camera hung, URL serving a cached frame, capture software
        # stopped writing) still classifies cleanly - the same frame, every pass - so
        # none of the failure paths fire, and an OPEN frame would be reported all
        # night. When the user opts in, report CLOSED once the image has not
        # changed for the stale threshold.
        # Applied after model_status so the toggle baseline stays the model's view,
        # and before the override, which still wins.
        stale_minutes = self._image_unchanged_minutes()
        if (self._stale_failsafe_enabled(config)
                and stale_minutes is not None
                and stale_minutes >= self._stale_threshold_minutes(config)):
            if final_status == "OPEN" and self.logger:
                self.logger.warning(
                    f"Image unchanged for {stale_minutes:.0f} min - reporting CLOSED "
                    f"instead of OPEN until the camera feed recovers")
            final_status = "CLOSED"
            override_reason = f" (Image unchanged for {stale_minutes:.0f} min - failsafe)"

        # Apply the manual override last — it wins over both the model and the sun guard
        # for the reported roof status. (The ASCOM IsSafe flag keeps its own sun check.)
        manual_override = self.get_manual_override()
        if manual_override:
            if manual_override != final_status and self.logger:
                self.logger.warning(
                    f"MANUAL OVERRIDE: reporting {manual_override} instead of {final_status} "
                    f"(model said {image_status})"
                )
            final_status = manual_override
            override_reason = f" (Manual override: {manual_override})"

        # Log the analysis
        now = datetime.now().strftime("%Y-%m-%d %I:%M:%S%p")
        sun_angle = self.calculate_sun_angle(config)

        log_message = f"Image: {latest}, Raw prediction: {image_status}, Final status: {final_status}"
        log_message += (", Sun angle: unknown" if sun_angle is None
                        else f", Sun angle: {sun_angle:.1f}°")
        
        if secondary_status:
            log_message += f", Secondary source: {secondary_status} (updated: {secondary_time})"
        else:
            log_message += ", Secondary source: Not available"
        
        if self.logger:
            self.logger.info(log_message)

        # ── Model vs secondary roof file comparison ───────────────────────────
        # Compare the raw model estimate (image_status) against the secondary roof
        # status file. Log loudly on mismatch and optionally capture the first frame
        # of each disagreement episode for manual review.
        if secondary_status:
            if image_status != secondary_status:
                if self.logger:
                    self.logger.warning(
                        f"MODEL/SECONDARY MISMATCH: model={image_status} (final={final_status}) "
                        f"!= secondary={secondary_status} (updated: {secondary_time}) for image {latest}"
                    )
                if config['save_on_disagreement'] and not self._in_disagreement:
                    self._save_frame_for_review(img_path, "disagree", config)
                self._in_disagreement = True
            else:
                self._in_disagreement = False

        # ── State toggle capture ──────────────────────────────────────────────
        # When the model's own status flips, save the frame so a wrongly-classified
        # transition can be reviewed/labelled later. This tracks model_status, not the
        # reported status: an override being applied or cleared is not a model toggle,
        # and must not advance the baseline either.
        if (config['save_on_toggle']
                and self.previous_classified_status is not None
                and model_status != self.previous_classified_status):
            self._save_frame_for_review(img_path, "toggle", config)
        self.previous_classified_status = model_status

        written = self._write_status_file(
            final_status, override_reason, config['output_path'], now)

        if not written:
            # The file on disk still shows the previous status. Publishing this
            # result to the ASCOM monitor anyway would let IsSafe and the roof
            # status file disagree, so report the pass as failed instead.
            return None, "Could not write roof status file"

        self._last_status_reason = override_reason.strip(" ()")
        self._last_classification = (latest, final_status, datetime.now(timezone.utc))
        self._last_good_pass_at = self._last_classification[2]
        if self._failsafe_active:
            self._failsafe_active = False
            if self.logger:
                self.logger.warning("Classification recovered - fail-safe status lifted")

        print(f"[{final_status}] {latest}")
        return latest, final_status

    def _write_status_file(self, status, reason="", output_path=None, timestamp=None):
        """Write the roof status file as a single line.

        Format follows the SRO Roof File spec:
        https://interactiveastronomy.com/skyroof_help/SROrooffile.html
        """
        if output_path is None:
            output_path = self.output_path.get()
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %I:%M:%S%p")
        line = format_status_line(status, reason, timestamp)
        try:
            # Written atomically: ASCOM clients and SkyRoof poll this file, and a
            # plain truncate-then-write lets them read an empty or partial line.
            atomic_write_text(output_path, line, allow_in_place_fallback=True)
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Could not write status file {output_path}: {e}")
            return False

    def update_monitoring_status(self, filename, status):
        """Update the monitoring status display"""
        if filename and status:
            checked = datetime.now().strftime("%H:%M:%S")
            self.status_label.config(
                text=f"Last check at {checked}: {self._short_source(filename)}", fg=COLOR_TEXT)
            self._update_secondary_status_display()

            # Update hash/stale display and check for stale warning
            stale_warning = self._update_hash_status_display()
            self._update_sun_status_display()

            if stale_warning:
                self.statusbar_label.config(
                    text=f"● Monitoring: roof {status}, image not changing", fg=COLOR_WARN)
            else:
                self.statusbar_label.config(text=f"● Monitoring: roof {status}", fg=COLOR_OK)
        else:
            # *status* carries the reason the pass failed; show it rather than a
            # generic error, so a dead camera URL, an unreadable frame and a
            # missing folder can be told apart without opening the log.
            reason = status or "Error checking files"
            text = f"Monitoring: {reason}"
            if self._failsafe_active:
                text += " — status file set to CLOSED (fail-safe)"
            self.status_label.config(text=text, fg=COLOR_ERROR)
            self.statusbar_label.config(text="● Monitoring: last check failed", fg=COLOR_ERROR)
        self._render_roof_status()

    def _update_secondary_status_display(self):
        """Show the secondary roof status next to the model's, when one is configured."""
        if not hasattr(self, "secondary_status_label"):
            return
        if not self.secondary_source_enabled.get():
            self._set_secondary_line(None)
            return
        # The result from the classification pass that just ran, read on its
        # worker thread; reading again here could block the UI on a slow URL.
        secondary_status, secondary_time = getattr(self, "_last_secondary", None) or (None, None)
        if secondary_status:
            when = secondary_time.strftime("%H:%M UTC") if secondary_time else "unknown time"
            self._set_secondary_line(
                f"Secondary roof status: {secondary_status} (updated {when})", COLOR_TEXT)
        else:
            self._set_secondary_line("Secondary roof status: unavailable", COLOR_WARN)

    def _set_secondary_line(self, text, fg=COLOR_TEXT):
        """Show the secondary-status line with *text*, or remove it when None, so an
        unused line does not leave a gap in the status card."""
        label = self.secondary_status_label
        if text is None:
            label.pack_forget()
            return
        label.config(text=text, fg=fg)
        if not label.winfo_manager():
            label.pack(fill="x", before=self._autostart_check)

    def update_countdown(self, seconds_remaining):
        """Update the countdown display"""
        if seconds_remaining > 0:
            self.countdown_label.config(text=f"Next check in {seconds_remaining} s", fg=COLOR_MUTED)
        else:
            self.countdown_label.config(text="Checking now…", fg=COLOR_MUTED)

    def clear_monitoring_status(self):
        """Clear the monitoring status when stopped"""
        self.monitoring_active = False
        self.status_label.config(text="Not monitoring", fg=COLOR_MUTED)
        self.countdown_label.config(text="")
        self.statusbar_label.config(text="● Monitoring off", fg=COLOR_MUTED)
        self._sync_monitoring_controls()
        if hasattr(self, 'hash_status_label'):
            self.hash_status_label.config(text="Image: —", fg=COLOR_MUTED)
        if hasattr(self, 'secondary_status_label'):
            self._set_secondary_line(None)
        self._render_roof_status()

    # ── New feature helpers ───────────────────────────────────────────────────

    def _fetch_image_from_url(self, url):
        """Download an image from *url* to a temporary file. Returns the temp file path,
        or None if the download fails."""
        tmp_path = None
        try:
            suffix = ".jpg"
            for ext in (".png", ".jpg", ".jpeg"):
                if url.lower().split("?")[0].endswith(ext):
                    suffix = ext
                    break
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(tmp_fd)
            req = urllib.request.Request(url, headers={"User-Agent": "SyntheticRoofStatus/1.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read()
            with open(tmp_path, "wb") as f:
                f.write(data)
            return tmp_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to fetch image from URL {url}: {e}")
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            return None

    def _send_webhook(self, url, payload):
        """POST *payload* as JSON to *url*.

        Returns ``(ok, detail)``: *ok* is True only for a 2xx response, and
        *detail* describes the outcome ("HTTP 204", or the error). Failures are
        logged, never raised, so a dead webhook cannot disturb the monitor loop -
        but callers such as the Test button can still tell the user it failed.
        """
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                code = response.status
            detail = f"HTTP {code}"
            if 200 <= code < 300:
                if self.logger:
                    self.logger.info(f"Webhook sent to {url}: {detail}")
                return True, detail
        except Exception as e:
            detail = str(e)
        if self.logger:
            self.logger.error(f"Failed to send webhook to {url}: {detail}")
        return False, detail

    def _check_and_send_notifications(self, status, config=None):
        """Check notification conditions and fire webhooks as appropriate.

        Called from the monitor background thread after each classification cycle,
        so every setting comes from the *config* snapshot rather than from the Tk
        variables, which may only be read on the UI thread.
        """
        if config is None:
            config = self._get_monitor_config()
        if config is None:
            if self.logger:
                self.logger.error("Skipping notifications: no configuration available")
            return

        now = datetime.utcnow()
        ts = now.isoformat() + "Z"

        # Roof open transition notification
        if config.get('notif_open_enabled') and status == "OPEN" and self.previous_status != "OPEN":
            url = config.get('notif_open_url', '')
            if url:
                self._send_webhook(url, {"event": "roof_open", "status": status, "timestamp": ts})

        # Roof closed transition notification
        if config.get('notif_closed_enabled') and status == "CLOSED" and self.previous_status != "CLOSED":
            url = config.get('notif_closed_url', '')
            if url:
                self._send_webhook(url, {"event": "roof_closed", "status": status, "timestamp": ts})

        self.previous_status = status

        # Stale image notification.
        # The notification is sent once when the image first becomes stale, then re-sent
        # after every additional stale_minutes interval while the image remains unchanged.
        stale_minutes_val = self._stale_threshold_minutes(config)
        elapsed_minutes = self._image_unchanged_minutes(now)
        is_stale = elapsed_minutes is not None and elapsed_minutes >= stale_minutes_val
        elapsed_minutes = elapsed_minutes or 0.0

        if config.get('notif_stale_enabled') and is_stale:
            url = config.get('notif_stale_url', '')
            if url:
                # Re-send at most once per stale_minutes interval
                already_sent = (
                    self._last_stale_notification_time is not None
                    and (now - self._last_stale_notification_time).total_seconds() / 60.0 < stale_minutes_val
                )
                if not already_sent:
                    self._send_webhook(url, {
                        "event": "image_stale",
                        "status": status,
                        "stale_minutes": round(elapsed_minutes, 1),
                        "timestamp": ts,
                    })
                    self._last_stale_notification_time = now

        # Heartbeat notification — fires every heartbeat_minutes interval while monitoring is
        # active, but is suppressed whenever the image is stale.
        if config.get('notif_heartbeat_enabled') and not is_stale:
            url = config.get('notif_heartbeat_url', '')
            if url:
                try:
                    heartbeat_minutes = float(config.get('notif_heartbeat_minutes', 5))
                except (TypeError, ValueError):
                    heartbeat_minutes = 5.0
                interval_elapsed = (
                    self._last_heartbeat_time is None
                    or (now - self._last_heartbeat_time).total_seconds() / 60.0 >= heartbeat_minutes
                )
                if interval_elapsed:
                    self._send_webhook(url, {"event": "heartbeat", "status": status, "timestamp": ts})
                    self._last_heartbeat_time = now

    @staticmethod
    def _stale_threshold_minutes(config):
        """The configured stale-image threshold in minutes, or the default when unset
        or invalid. Used by the stale notification, the UI and the stale fail-safe."""
        try:
            minutes = float(config.get('notif_stale_minutes', DEFAULT_STALE_MINUTES))
        except (TypeError, ValueError):
            return DEFAULT_STALE_MINUTES
        if not math.isfinite(minutes) or minutes <= 0:
            return DEFAULT_STALE_MINUTES
        return minutes

    @staticmethod
    def _stale_failsafe_enabled(config):
        """True only when the user opted into reporting CLOSED for a stale frame.
        Missing or unrecognised values mean the default: keep the model's view."""
        return config.get('stale_image_action', DEFAULT_STALE_ACTION) == STALE_ACTION_CLOSED

    def _image_unchanged_minutes(self, now=None):
        """Minutes since the image hash last changed, or None before the first frame."""
        if self.last_new_hash_time is None:
            return None
        now = now or datetime.utcnow()
        return (now - self.last_new_hash_time).total_seconds() / 60.0

    def _update_hash_status_display(self):
        """Refresh the image hash status label. Returns True if the image is considered stale."""
        if not hasattr(self, "hash_status_label"):
            return False

        if self.last_new_hash_time is None:
            self.hash_status_label.config(text="Image: waiting for the first check", fg=COLOR_MUTED)
            return False

        elapsed_seconds = (datetime.utcnow() - self.last_new_hash_time).total_seconds()
        elapsed_str = self._format_elapsed(elapsed_seconds)

        stale_minutes = self._stale_threshold_minutes(
            {'notif_stale_minutes': self.notif_stale_minutes.get()})

        is_stale = elapsed_seconds / 60.0 >= stale_minutes

        if is_stale:
            self.hash_status_label.config(
                text=f"Image unchanged for {elapsed_str} (stale after {stale_minutes:g} min)",
                fg=COLOR_WARN,
            )
        else:
            self.hash_status_label.config(text=f"Image last changed {elapsed_str} ago", fg=COLOR_TEXT)
        return is_stale

    def _update_sun_status_display(self):
        """Refresh the Monitoring-tab sun altitude label, flagging when the sun is
        above the configured safe threshold (i.e. unsafe to report OPEN)."""
        if not hasattr(self, "sun_status_label"):
            return
        try:
            sun_angle = self.calculate_sun_angle()
            threshold = float(self.sun_angle_threshold.get())
        except Exception:
            sun_angle = None
            threshold = None

        if sun_angle is None or threshold is None:
            self.sun_status_label.config(
                text="Sun altitude unknown, so OPEN is not reported", fg=COLOR_WARN)
            return

        if sun_angle < threshold:
            self.sun_status_label.config(
                text=f"Sun at {sun_angle:.1f}°, below the {threshold:g}° limit", fg=COLOR_TEXT)
        else:
            self.sun_status_label.config(
                text=f"Sun at {sun_angle:.1f}°, above the {threshold:g}° limit, so OPEN is not reported",
                fg=COLOR_WARN,
            )

    # ── Manual override of the reported roof status ───────────────────────────

    def _compute_override_expiry(self, duration_label):
        """Return the local (naive) expiry time for *duration_label*, or None for 'Forever'."""
        now = datetime.now()
        if duration_label == "1 hour":
            return now + timedelta(hours=1)
        if duration_label == "4 hours":
            return now + timedelta(hours=4)
        if duration_label == "Until noon":
            noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
            if noon <= now:
                noon += timedelta(days=1)
            return noon
        if duration_label == "Until midnight":
            return (now.replace(hour=0, minute=0, second=0, microsecond=0)
                    + timedelta(days=1))
        return None  # "Forever"

    def _defer_to_ui(self, func):
        """Run *func* on the UI thread, immediately if already there."""
        if threading.current_thread() is threading.main_thread():
            func()
            return
        try:
            self.root.after(0, func)
        except Exception:
            pass  # window is closing

    def get_manual_override(self):
        """Return "OPEN"/"CLOSED" while a manual override is in force, else None.

        An expired override is cleared as a side effect. Safe to call from the monitor
        thread: the in-memory state is updated here, and everything that touches Tk —
        including save_settings(), which reads every Tk variable — is deferred to the
        UI thread.
        """
        if self.override_active is None:
            return None
        if self.override_expiry is not None and datetime.now() >= self.override_expiry:
            expired = self.override_active
            self.override_active = None
            self.override_expiry = None
            # The cached result was produced under the override; it says nothing
            # about what the model sees now.
            self._last_classification = None
            if self.logger:
                self.logger.warning(
                    f"Manual override ({expired}) expired — reverting to model output"
                )
            self._defer_to_ui(self._finalize_override_change)
            return None
        return self.override_active

    def _finalize_override_change(self):
        """Persist the override state and resync the widgets (UI thread only)."""
        self._sync_override_ui()
        self.save_settings()

    def _sync_override_ui(self):
        """Bring the override radio buttons and label back in line with the committed
        override state (UI thread only)."""
        self.override_mode.set(self.override_active or "AUTO")
        self._update_override_display()

    def apply_manual_override(self):
        """Commit the staged radio-button selection as the active override."""
        mode = self.override_mode.get()
        if mode == "AUTO":
            self.clear_manual_override()
            return

        duration = self.override_duration.get()
        self.override_active = mode
        self.override_expiry = self._compute_override_expiry(duration)
        self.save_settings()

        until = self.override_expiry.strftime("%Y-%m-%d %H:%M local") if self.override_expiry else "forever"
        if self.logger:
            self.logger.warning(f"MANUAL OVERRIDE APPLIED: reporting {mode} until {until}")
        self._update_override_display()

        # Write the forced status straight away. Waiting for the next monitoring cycle
        # would leave the status file — and therefore ASCOM clients — reporting the
        # previous result for up to a minute, or indefinitely if monitoring is stopped.
        written = self._write_status_file(mode, f" (Manual override: {mode})")
        # Same for the ASCOM flag, which would otherwise wait for its next refresh.
        self._refresh_ascom_safety()

        self._render_roof_status()
        if not written:
            messagebox.showwarning(
                "Status File Not Updated",
                f"The override is active and ASCOM clients see {mode}, but the roof status "
                f"file could not be written:\n\n{self.output_path.get()}\n\n"
                "Check the output path on the Configuration tab.")

    def clear_manual_override(self):
        """Drop any active override and return to model-driven status."""
        was = self.override_active
        self.override_active = None
        self.override_expiry = None
        if was:
            # The cached result carries the forced status. Drop it so ASCOM reports
            # "no status yet" (unsafe) until the re-classification below lands,
            # rather than the override that was just cleared.
            self._last_classification = None
        self.override_mode.set("AUTO")
        self.save_settings()
        if was and self.logger:
            self.logger.warning(f"Manual override ({was}) cleared — reverting to model output")
        self._update_override_display()
        self._render_roof_status()
        if was:
            self._refresh_ascom_safety()
            # Re-classify now so the status file stops reporting the cleared override
            self._refresh_status_now()

    def _refresh_status_now(self):
        """Re-run classification in the background so the status file reflects the
        current model output without waiting for the next monitoring cycle."""
        if not self.model:
            if self.logger:
                self.logger.warning(
                    "Override cleared but no model is loaded — the status file still holds "
                    "the last written status"
                )
            return

        config = self._snapshot_monitor_config()

        def work():
            try:
                # Force a fresh pass: any cached result predates the override change.
                filename, status = self.classify_latest_png(config, max_cache_age=0)
                self._refresh_ascom_safety()
                self.root.after(0, lambda f=filename, s=status: self.update_monitoring_status(f, s))
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error refreshing status after override change: {e}")

        threading.Thread(target=work, daemon=True).start()

    def _refresh_ascom_safety(self):
        """Recompute the ASCOM IsSafe flag now instead of on its next timed refresh.

        Called whenever the reported status changes, so a client polling IsSafe
        does not act on a flag up to UPDATE_INTERVAL_SECONDS out of date.
        """
        server = self.ascom_server
        if server is None:
            return
        try:
            server.refresh_safety_status()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Could not refresh ASCOM safety status: {e}")

    def _format_override_remaining(self):
        """Human-readable time left on the current override."""
        if self.override_expiry is None:
            return "no expiry"
        remaining = (self.override_expiry - datetime.now()).total_seconds()
        if remaining <= 0:
            return "expiring"
        hours, rem = divmod(int(remaining), 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes}m left"
        if minutes:
            return f"{minutes}m {seconds}s left"
        return f"{seconds}s left"

    def _update_override_display(self):
        """Refresh the Monitoring-tab override status label."""
        if not hasattr(self, "override_status_label"):
            return
        clear_btn = getattr(self, "override_clear_btn", None)
        if clear_btn is not None:
            self._set_enabled(clear_btn, self.override_active is not None)
        if self.override_active is None:
            self.override_status_label.config(
                text="No override. The model's classification is reported.", fg=COLOR_MUTED,
                font=self.font_small)
            return

        if self.override_expiry is None:
            detail = "until you clear it"
        else:
            when = self.override_expiry.strftime(
                "%H:%M" if self.override_expiry.date() == datetime.now().date() else "%a %H:%M")
            detail = f"until {when} ({self._format_override_remaining()})"
        self.override_status_label.config(
            text=f"Override active: reporting {self.override_active} {detail}",
            fg=COLOR_WARN, font=self.font_bold,
        )

    def _tick_override_display(self):
        """Once-a-second refresh: keeps the override countdown live, notices expiry
        even when monitoring is not running, and keeps the status displays current."""
        try:
            self.get_manual_override()  # clears the override if it has expired
            self._update_override_display()
            self._render_roof_status()
            self._update_ascom_display()
        finally:
            try:
                self.root.after(1000, self._tick_override_display)
            except Exception:
                pass  # window is closing

    # ── Latest-image preview (Monitoring tab) ─────────────────────────────────

    def _resolve_latest_image(self, camera_url=None, folder=None):
        """Locate the newest image to classify or preview.

        Returns (path, caption, is_temp, error). *path* is None when no image is
        available, in which case *error* describes why. When *is_temp* is True the
        caller owns the temporary file and must delete it.

        *camera_url* and *folder* may be supplied by a caller that already read them
        on the UI thread; otherwise they are read from the Tk variables here.
        """
        camera_url = (self.camera_url.get() if camera_url is None else camera_url).strip()
        if camera_url:
            tmp_path = self._fetch_image_from_url(camera_url)
            if tmp_path is None:
                return None, None, False, "Failed to fetch image from URL"
            return tmp_path, camera_url, True, None

        folder = self.monitor_path.get() if folder is None else folder
        if not folder:
            return None, None, False, "No image folder or camera URL set"
        if not os.path.isdir(folder):
            return None, None, False, f"Image folder not found: {folder}"

        try:
            names = os.listdir(folder)
        except OSError as e:
            return None, None, False, f"Could not list folder: {e}"

        # Cameras and cleanup scripts delete old frames while we look, so a file
        # can vanish between listdir and getmtime; skip it rather than raising.
        images = []
        for name in names:
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            try:
                images.append((os.path.getmtime(os.path.join(folder, name)), name))
            except OSError:
                continue
        if not images:
            return None, None, False, "No image files found"

        _mtime, latest = max(images)
        return os.path.join(folder, latest), latest, False, None

    def on_preview_enabled_changed(self):
        """Show or hide the preview panel (and stop doing the work when hidden)."""
        self._preview_on = self.preview_enabled.get()
        if self._preview_on:
            self.preview_holder.pack(side=tk.TOP, fill="both", expand=True)
            self.preview_refresh_btn.state(["!disabled"])
            self.refresh_preview()
        else:
            self.preview_holder.pack_forget()
            self.preview_refresh_btn.state(["disabled"])
            self.preview_caption_label.config(text="Preview hidden", fg=COLOR_MUTED)

    def _capture_preview(self, img_path, caption):
        """Scale *img_path* for the Monitoring-tab preview and hand it to the UI thread.

        Safe to call from the monitor thread — the OpenCV work happens here and all Tk
        work is deferred with root.after.
        """
        # _preview_on mirrors the checkbox as a plain bool: this runs on the monitor
        # thread, which must not touch Tk variables.
        if not hasattr(self, "preview_label") or not self._preview_on:
            return
        tmp_path = None
        try:
            img = cv2.imread(img_path)
            if img is None:
                return
            h, w = img.shape[:2]
            max_w, max_h = getattr(self, "_preview_max", (_PREVIEW_IMG_MAX_W, _PREVIEW_IMG_MAX_H))
            scale = min(max_w / w, max_h / h, 1.0)
            disp = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(tmp_fd)
            cv2.imwrite(tmp_path, disp)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not build preview image for {img_path}: {e}")
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            return

        try:
            self.root.after(0, self._apply_preview, tmp_path, caption)
        except Exception:
            # The window is going away — drop the prepared image
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _apply_preview(self, tmp_path, caption):
        """Display a prepared preview image (UI thread only)."""
        if not hasattr(self, "preview_label"):
            return
        try:
            tk_img = tk.PhotoImage(file=tmp_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return

        previous = self._preview_tmp_path
        self._preview_tk_img = tk_img          # keep a reference alive for Tk
        self._preview_tmp_path = tmp_path
        # width/height are character cells for a text label but pixels for an image
        # label, so they must be restated in pixels or the image is clipped.
        self.preview_label.config(image=tk_img, text="",
                                  width=tk_img.width(), height=tk_img.height())
        self.preview_caption_label.config(
            text=f"{self._short_source(caption, 56)}  ·  {datetime.now().strftime('%H:%M:%S')}",
            fg=COLOR_MUTED,
        )

        if previous and previous != tmp_path:
            try:
                os.unlink(previous)
            except OSError:
                pass

    def refresh_preview(self):
        """Fetch and display the latest image on demand (button handler)."""
        if self._preview_busy:
            return
        self._preview_busy = True
        self.preview_caption_label.config(text="Loading the latest image…", fg=COLOR_MUTED)

        # Read the Tk variables here, on the UI thread, and hand the plain strings to
        # the worker — Tk variables must not be touched from another thread.
        camera_url = self.camera_url.get()
        folder = self.monitor_path.get()

        def do_refresh():
            try:
                img_path, caption, is_temp, error = self._resolve_latest_image(camera_url, folder)
                if img_path is None:
                    try:
                        self.root.after(
                            0,
                            lambda: self.preview_caption_label.config(
                                text=f"No preview: {error}", fg=COLOR_MUTED
                            ),
                        )
                    except Exception:
                        pass
                    return
                try:
                    self._capture_preview(img_path, caption)
                finally:
                    if is_temp:
                        try:
                            os.unlink(img_path)
                        except OSError:
                            pass
            finally:
                self._preview_busy = False

        threading.Thread(target=do_refresh, daemon=True).start()

    def _test_secondary_source(self):
        """Test reading the configured secondary roof status file or URL."""
        source = self.secondary_source_path.get().strip()
        if not source:
            messagebox.showwarning("No Source", "Please enter a secondary roof status file path or URL first.")
            return

        def do_test():
            # Bypass the cache so the test always hits the real source, and read it
            # with the enabled flag forced on without touching Tk from this thread.
            self._secondary_cache = None
            try:
                status, mod_time = self._read_secondary_values(True, source)
            finally:
                self._secondary_cache = None

            if status:
                time_str = mod_time.strftime("%Y-%m-%d %H:%M:%S UTC") if mod_time else "unknown"
                msg = f"Read status: {status}\nLast updated: {time_str}\n\nSource: {source}"
                self.root.after(0, lambda m=msg: messagebox.showinfo("Secondary Source OK", m))
            else:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Secondary Source Failed",
                        f"Could not read an OPEN/CLOSED status from:\n{source}\n\n"
                        "Check the path/URL and that the last line contains OPEN or CLOSED.\n"
                        "Enable logging for the detailed error.",
                    ),
                )

        threading.Thread(target=do_test, daemon=True).start()

    def _test_camera_url(self):
        """Test downloading an image from the configured camera URL."""
        url = self.camera_url.get().strip()
        if not url:
            messagebox.showwarning("No URL", "Please enter a camera image URL first.")
            return

        def do_test():
            tmp_path = self._fetch_image_from_url(url)
            if tmp_path:
                try:
                    img = cv2.imread(tmp_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        msg = f"Successfully downloaded image from URL.\nSize: {w}×{h} pixels"
                    else:
                        msg = ("Downloaded file but could not parse it as an image.\n"
                               "Check that the URL points to a valid image file.")
                    os.unlink(tmp_path)
                    self.root.after(0, lambda m=msg: messagebox.showinfo("URL Test Success", m))
                except Exception as e:
                    self.root.after(0, lambda e=e: messagebox.showerror("URL Test Error", f"Error: {e}"))
            else:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "URL Test Failed",
                        f"Failed to download image from:\n{url}\n\nCheck the URL and network connectivity.",
                    ),
                )

        threading.Thread(target=do_test, daemon=True).start()

    def _test_webhook(self, url_var):
        """Send a test POST to the webhook URL stored in *url_var*."""
        url = url_var.get().strip()
        if not url:
            messagebox.showwarning("No URL", "Please enter a webhook URL first.")
            return

        payload = {
            "event": "test",
            "status": "TEST",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": "Test notification from Synthetic RoofStatusFile.txt",
        }

        def do_send():
            # _send_webhook never raises, so its result is the only way to know
            # whether the POST landed. Ignoring it reported "sent" for every
            # typo'd URL and 500 response.
            ok, detail = self._send_webhook(url, payload)
            if ok:
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Webhook Sent",
                        f"Test webhook sent to:\n{url}\n\nServer replied {detail}. "
                        "Check the destination for the payload.",
                    ),
                )
            else:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Webhook Error", f"Test webhook to:\n{url}\n\nfailed: {detail}"),
                )

        threading.Thread(target=do_send, daemon=True).start()

    def monitor_loop(self, stop_event=None):
        """Classify every MONITOR_INTERVAL_SECONDS until *stop_event* is set.

        Runs on its own thread for the whole night, so no single pass is allowed
        to end it: an exception is logged, the pass counts as failed, and the
        loop carries on. (It used to propagate and kill the thread silently,
        leaving the status file frozen on its last line while the UI still said
        "Monitoring: Active".)
        """
        if stop_event is None:
            stop_event = self._monitor_stop_event
        if stop_event is None:
            # Only start_monitoring creates a run; without one there is nothing
            # that could ever stop this loop.
            if self.logger:
                self.logger.error("monitor_loop called without a monitoring run - not starting")
            return
        try:
            while not stop_event.is_set():
                config = None
                try:
                    # Take a fresh configuration snapshot on the UI thread each
                    # cycle, so the worker never touches Tk variables but still
                    # sees setting changes.
                    config = self._request_monitor_config()
                    if config is None:
                        if self.logger:
                            self.logger.error("Skipping monitoring cycle: no configuration available")
                        filename, status = None, "No configuration available"
                    else:
                        filename, status = self.classify_latest_png(config, max_cache_age=0)
                except Exception as e:
                    if self.logger:
                        self.logger.exception(f"Monitoring pass failed: {e}")
                    filename, status = None, f"Error: {e}"

                if stop_event.is_set():
                    break  # stopped mid-pass; do not report into a newer run's UI

                if filename is None:
                    try:
                        self._apply_failsafe_status(config)
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Could not apply fail-safe status: {e}")

                # Publish the new result (or its absence) to ASCOM clients now.
                self._refresh_ascom_safety()

                self._defer_to_ui(lambda f=filename, s=status: self.update_monitoring_status(f, s))

                # Send notifications if configured (runs in background thread)
                if status in ("OPEN", "CLOSED") and config is not None:
                    try:
                        self._check_and_send_notifications(status, config)
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Error sending notifications: {e}")

                # Count down to a real deadline, one UI update per tick, so the
                # interval holds whatever the tick length; a stop request ends
                # the wait at once.
                deadline = time.monotonic() + MONITOR_INTERVAL_SECONDS
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._defer_to_ui(lambda r=math.ceil(remaining): self.update_countdown(r))
                    if stop_event.wait(min(_COUNTDOWN_TICK_SECONDS, remaining)):
                        break
        finally:
            self._defer_to_ui(lambda: self._on_monitor_loop_exit(stop_event))

    def _on_monitor_loop_exit(self, stop_event):
        """Reset the monitoring UI when a loop ends - unless a newer run has
        started in the meantime, whose state is not this loop's to clear."""
        if self._monitor_stop_event is stop_event:
            self.clear_monitoring_status()

    def _apply_failsafe_status(self, config=None, now=None):
        """Write a fail-safe status line once classification has failed for too long.

        A failed pass leaves the status file untouched, which is right for a
        one-off glitch. But when the camera feed dies, the file would otherwise
        keep saying OPEN for the rest of the night with nothing to tell a reader
        that it is stale. After FAILSAFE_AFTER_SECONDS without a good pass it is
        rewritten as CLOSED, the same point at which the ASCOM flag already turns
        unsafe. An active manual override still wins: that is the tool for "the
        camera is obscured and I know better".

        Returns True when a fail-safe line was written.
        """
        with self._classify_lock:
            now = now or datetime.now(timezone.utc)
            last_good = self._last_good_pass_at
            if last_good is not None and (now - last_good).total_seconds() <= FAILSAFE_AFTER_SECONDS:
                return False

            output_path = (config or self._last_monitor_config or {}).get('output_path')
            if not output_path:
                if self.logger:
                    self.logger.error("Cannot write fail-safe status: no output path configured")
                return False

            override = self.get_manual_override()
            if override:
                status, reason = override, f" (Manual override: {override})"
            else:
                status, reason = "CLOSED", " (No valid classification - failsafe)"

            # Stamp the line with the same instant the age check used, in the
            # local time the status file is written in.
            timestamp = now.astimezone().strftime("%Y-%m-%d %I:%M:%S%p")
            written = self._write_status_file(status, reason, output_path, timestamp)
            if written:
                # _failsafe_active means "the file holds the fail-safe CLOSED
                # line". An override line is not that, even when written here,
                # so it must not trigger the fail-safe UI note or the later
                # "recovered" log. A failed write leaves the file, and the flag,
                # as they were.
                if override:
                    self._failsafe_active = False
                else:
                    if not self._failsafe_active and self.logger:
                        self.logger.warning(
                            f"No valid classification for over {FAILSAFE_AFTER_SECONDS}s - "
                            f"status file set to CLOSED until classification recovers")
                    self._failsafe_active = True
            return written

    def toggle_monitoring(self):
        """Toggle monitoring on or off from the status bar button"""
        if self.monitoring_active:
            self.stop_monitoring()
            return
        # Checked only for a manual start. An auto-start at boot must still run
        # when, say, a network share is not mounted yet: every pass then fails and
        # the fail-safe reports CLOSED until the folder appears.
        if self.model and not self.camera_url.get().strip() and not os.path.isdir(self.monitor_path.get()):
            messagebox.showerror(
                "No Image Source",
                "Choose the folder your camera saves images to, or a camera image URL, "
                "on the Configuration tab.")
            self._select_tab("config")
            return
        self.start_monitoring()

    def start_monitoring(self):
        if not self.model:
            messagebox.showerror("No Model Loaded",
                                 "Train a model or load one on the Training & Model tab "
                                 "before starting monitoring.")
            self._select_tab("training")
            return

        if self.monitoring_active:
            return

        # Validate configuration
        try:
            float(self.latitude.get())
            float(self.longitude.get())
            float(self.sun_angle_threshold.get())
        except ValueError:
            messagebox.showerror("Invalid Observatory Settings",
                                 "Latitude, longitude and the sun limit must be numbers. "
                                 "Check them on the Configuration tab.")
            self._select_tab("config")
            return
        
        # Setup logging with current settings
        self.setup_logging()
        
        # Warn if the classifier log file is the same as the roof status output file.
        # This would cause diagnostic log entries to corrupt the roof status file format.
        if self.log_enabled.get() and self._log_path_conflicts_with_output():
            self._warn_log_path_conflict("Please set a different path for the classifier log file in the Configuration tab.")
        
        stop_event = threading.Event()
        self._monitor_stop_event = stop_event
        self.monitoring_active = True
        # Grace period for the fail-safe starts now, not at the last run's success
        self._last_good_pass_at = datetime.now(timezone.utc)
        self._failsafe_active = False
        # Reset notification state so transition and stale notifications work correctly
        self.previous_status = None
        self._last_stale_notification_time = None
        self._last_heartbeat_time = None
        self.last_image_hash = None
        self.last_new_hash_time = None
        self.previous_classified_status = None
        self._in_disagreement = False
        self._last_classification = None
        self.status_label.config(text="Starting…", fg=COLOR_MUTED)
        self.countdown_label.config(text="")
        self.statusbar_label.config(text="● Monitoring: starting", fg=COLOR_OK)
        self._sync_monitoring_controls()
        self._update_sun_status_display()
        self._render_roof_status()
        
        if self.logger:
            self.logger.info("Monitoring started")
            self.logger.info(f"Observatory location: {self.latitude.get()}°, {self.longitude.get()}°")
            self.logger.info(f"Sun angle threshold: {self.sun_angle_threshold.get()}°")
            if self.secondary_source_enabled.get():
                self.logger.info(f"Secondary source enabled: {self.secondary_source_path.get()}")
            else:
                self.logger.info("Secondary source disabled")
        
        threading.Thread(target=self.monitor_loop, args=(stop_event,), daemon=True).start()

    def stop_monitoring(self):
        """Stop the current run. The UI is reset at once; a pass already in
        progress finishes in the background but reports nothing further."""
        if self._monitor_stop_event is not None:
            self._monitor_stop_event.set()
        self.clear_monitoring_status()
        if self.logger:
            self.logger.info("Monitoring stopped")

    def save_current_model_as(self):
        """Save the currently loaded model to a new location"""
        if not self.model:
            messagebox.showerror("No Model Loaded", "Train or load a model before saving it.")
            return
            
        current_path = self.model_path.get()
        initial_dir = os.path.dirname(current_path) if current_path else os.getcwd()
        path = filedialog.asksaveasfilename(
            title="Save Model As",
            defaultextension=".joblib", 
            filetypes=[("Joblib model", "*.joblib")],
            initialdir=initial_dir
        )
        if path:
            try:
                dump(self.model, path)
            except Exception as e:
                messagebox.showerror("Could Not Save Model", f"Failed to save model: {e}")
                return
            # The saved file is now the current model: it is reloaded on startup
            # and the next training run saves over it.
            self.model_path.set(path)
            self.save_settings()
            self._update_model_display()
            self._set_activity(f"Model saved to {path}.")

    def save_model_as(self):
        """Legacy method - redirects to save_current_model_as for compatibility"""
        self.save_current_model_as()

    def browse_output_file(self):
        """Browse for output file path"""
        current_path = self.output_path.get()
        initial_dir = os.path.dirname(current_path) if current_path else os.getcwd()
        path = filedialog.asksaveasfilename(
            title="Select Output Status File Location",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=initial_dir,
            initialfile=os.path.basename(current_path) if current_path else "RoofStatusFile.txt"
        )
        if path:
            self.output_path.set(path)
            self.save_settings()

    def browse_monitor_folder(self):
        """Browse for monitor folder"""
        current_path = self.monitor_path.get()
        initial_dir = current_path if current_path and os.path.isdir(current_path) else os.getcwd()
        folder = filedialog.askdirectory(
            title="Select Folder to Monitor",
            initialdir=initial_dir
        )
        if folder:
            self.monitor_path.set(folder)
            self.save_settings()

    def browse_training_data_folder(self):
        """Browse for the base training data folder"""
        current = self.training_data_folder.get().strip()
        initial_dir = current if current and os.path.isdir(current) else os.getcwd()
        folder = filedialog.askdirectory(
            title="Select Training Data Folder",
            initialdir=initial_dir
        )
        if folder:
            self.training_data_folder.set(folder)
            self.save_settings()
            self.update_training_stats()

    def browse_validation_set(self):
        """Browse for a fixed validation set folder."""
        current = self.validation_set_path.get().strip()
        initial_dir = current if current and os.path.isdir(current) else os.getcwd()
        folder = filedialog.askdirectory(
            title="Select Fixed Validation Set Folder (must contain open/ and/or closed/ subfolders)",
            initialdir=initial_dir
        )
        if folder:
            self.validation_set_path.set(folder)
            self.save_settings()

    def _load_validation_data(self, folder):
        """Load images from open/ and closed/ subfolders of *folder*.

        Returns (X, y, file_names) where X is a list of flattened image arrays,
        y is a list of int labels (1=open, 0=closed), and file_names are relative paths.
        Raises ValueError with a human-readable message when no images can be loaded.
        """
        if not folder or not os.path.isdir(folder):
            raise ValueError("Validation set folder not found.")

        open_folder = os.path.join(folder, "open")
        closed_folder = os.path.join(folder, "closed")
        if not os.path.isdir(open_folder) and not os.path.isdir(closed_folder):
            raise ValueError("Validation folder must contain 'open' and/or 'closed' subfolders.")

        X, y, file_names = [], [], []
        for label, val in [("open", 1), ("closed", 0)]:
            sub = os.path.join(folder, label)
            if not os.path.isdir(sub):
                continue
            for file in sorted(os.listdir(sub)):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        img = self.prep_image(os.path.join(sub, file))
                        X.append(img.flatten())
                        y.append(val)
                        file_names.append(f"{label}/{file}")
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

        if not X:
            raise ValueError("No validation images found in the selected folder.")

        return X, y, file_names

    def benchmark_models(self):
        """Run one or more .joblib models against the fixed validation set and compare results."""
        val_folder = self.validation_set_path.get().strip()
        if not val_folder:
            messagebox.showerror(
                "No Validation Set",
                "Choose a fixed validation set folder on the Training & Model tab first."
            )
            return

        try:
            X_val, y_val, file_names = self._load_validation_data(val_folder)
        except ValueError as e:
            messagebox.showerror("Validation Set Error", str(e))
            return

        model_files = filedialog.askopenfilenames(
            title="Select one or more model files to benchmark",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")]
        )
        if not model_files:
            return

        X_arr = np.array(X_val)
        results = []
        errors = []
        for path in model_files:
            try:
                mdl = self._load_model_file(path)
                y_pred = mdl.predict(X_arr)
                acc = accuracy_score(y_val, y_pred)
                cm = confusion_matrix(y_val, y_pred)
                results.append((os.path.basename(path), acc, cm, y_pred, path))
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")

        if not results and errors:
            messagebox.showerror("Benchmark Error", "\n".join(errors))
            return

        self._show_benchmark_results(results, y_val, file_names, errors, val_folder)

    def _show_benchmark_results(self, results, y_true, file_names, errors, folder=None):
        """Display a side-by-side benchmark comparison in a Toplevel window."""
        win = self._make_dialog("Model Benchmark", min_size=(560, 420))
        body = ttk.Frame(win, padding=self._px(14))
        body.pack(fill="both", expand=True)

        ttk.Label(body, text=f"{len(results)} model{'s' if len(results) != 1 else ''} "
                             f"against {len(y_true)} validation images",
                  style="Heading.TLabel").pack(anchor="w", pady=(0, self._px(8)))

        # ── Summary, best first ───────────────────────────────────────────────
        summary_holder = ttk.Frame(body, height=self._px(22) * (min(len(results), 6) + 2))
        summary_holder.pack(fill="x")
        summary_holder.pack_propagate(False)
        summary = self._scrolled_tree(summary_holder, [
            ("model", "Model", 260), ("accuracy", "Accuracy", 90),
            ("false_open", "Wrongly OPEN", 110), ("false_closed", "Wrongly CLOSED", 110)],
            scroll_y=len(results) > 6)
        summary.tag_configure("best", font=self.font_bold)
        best_acc = max(r[1] for r in results) if results else 0.0
        for name, acc, _cm, y_pred, _path in sorted(results, key=lambda r: -r[1]):
            false_open = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            false_closed = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
            summary.insert("", tk.END, tags=("best",) if acc == best_acc else (),
                           values=(name, f"{acc * 100:.1f}%", false_open, false_closed))
        self._hint(body, "Wrongly OPEN is the costly mistake: the roof reported open while it "
                         "is closed.", pady=(self._px(4), 0))

        if errors:
            tk.Label(body, text="Could not run: " + "; ".join(errors), fg=COLOR_ERROR,
                     anchor="w", justify=tk.LEFT, wraplength=self._px(640)).pack(
                fill="x", pady=(self._px(6), 0))

        # ── Per-image detail ──────────────────────────────────────────────────
        controls = ttk.Frame(body)
        controls.pack(fill="x", pady=(self._px(14), self._px(6)))
        ttk.Label(controls, text="Per image", style="Heading.TLabel").pack(side=tk.LEFT)
        only_disputed = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Only images some model got wrong",
                        variable=only_disputed, command=lambda: fill()).pack(
            side=tk.LEFT, padx=(self._px(14), 0))
        if folder and hasattr(os, "startfile"):
            ttk.Label(controls, text="Double-click an image to open it.",
                      style="Muted.TLabel").pack(side=tk.RIGHT)

        columns = [("file", "Image", 240), ("actual", "Actual", 80)]
        columns += [(f"m{i}", r[0], 120) for i, r in enumerate(results)]
        detail = self._scrolled_tree(body, columns, scroll_x=len(results) > 3)
        detail.tag_configure("disputed", foreground=COLOR_ERROR)
        self._open_image_on_double_click(detail, folder)

        def fill():
            detail.delete(*detail.get_children())
            for i, (fname, true_val) in enumerate(zip(file_names, y_true)):
                preds = []
                any_wrong = False
                for _, _, _, y_pred, _ in results:
                    wrong = y_pred[i] != true_val
                    any_wrong = any_wrong or wrong
                    label = "OPEN" if y_pred[i] == 1 else "CLOSED"
                    preds.append(f"{label}  (wrong)" if wrong else label)
                if only_disputed.get() and not any_wrong:
                    continue
                detail.insert("", tk.END, tags=("disputed",) if any_wrong else (),
                              values=[fname, "OPEN" if true_val == 1 else "CLOSED"] + preds)

        fill()
        self._center_over_root(win, self._px(760), self._px(560))

    def _save_frame_for_review(self, img_path, reason, config=None):
        """Copy *img_path* into the unclassified folder so it can be labelled later.

        *reason* is a short tag (e.g. 'toggle', 'disagree') prepended to the filename
        so the cause of capture is visible in the Classify Images window.
        *config* is an optional snapshot from _get_monitor_config (see above).
        """
        try:
            if config is None:
                config = self._get_monitor_config() or {}
            unclassified_folder = self._get_training_class_folder(
                "unclassified", config.get('training_data_folder', ''))
            os.makedirs(unclassified_folder, exist_ok=True)
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            dest = os.path.join(unclassified_folder, f"{reason}_{stamp}_{os.path.basename(img_path)}")
            base, ext = os.path.splitext(dest)
            counter = 1
            while os.path.exists(dest):
                dest = f"{base}_{counter}{ext}"
                counter += 1
            shutil.copy2(img_path, dest)
            if self.logger:
                self.logger.info(f"Saved frame for review ({reason}): {dest}")
            self.root.after(0, self.update_training_stats)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving frame for review ({reason}): {e}")

    def save_sample_if_needed(self, img_path, config=None):
        """Randomly copy an image to the unclassified folder when sampling mode is active.

        *config* is an optional snapshot from _get_monitor_config (see above).
        """
        if config is None:
            config = self._get_monitor_config() or {}
        if not config.get('sample_mode_enabled'):
            return
        try:
            rate = float(config.get('sample_rate'))
        except (TypeError, ValueError):
            return
        if not 0.0 <= rate <= 1.0:
            return
        if random.random() >= rate:
            return
        try:
            unclassified_folder = self._get_training_class_folder(
                "unclassified", config.get('training_data_folder', ''))
            os.makedirs(unclassified_folder, exist_ok=True)
            dest = os.path.join(unclassified_folder, os.path.basename(img_path))
            base, ext = os.path.splitext(dest)
            counter = 1
            while os.path.exists(dest):
                dest = f"{base}_{counter}{ext}"
                counter += 1
            shutil.copy2(img_path, dest)
            if self.logger:
                self.logger.info(f"Saved random sample to unclassified folder: {dest}")
            # Refresh stats on the main thread
            self.root.after(0, self.update_training_stats)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving random sample: {e}")

    def open_classify_images_window(self):
        """Open the manual classification window for unclassified images."""
        unclassified_folder = self._get_training_class_folder("unclassified")
        if not os.path.isdir(unclassified_folder):
            messagebox.showinfo(
                "Nothing to Review",
                "There is no unclassified/ folder yet.\n\n"
                "Turn on one of the options under Collect Frames While Monitoring, or put "
                "images in an unclassified/ subfolder of the training data folder."
            )
            return

        images = sorted(
            [f for f in os.listdir(unclassified_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        if not images:
            messagebox.showinfo("Nothing to Review", "All unclassified images have been reviewed.")
            return

        self._show_classify_window(unclassified_folder, images)

    def _show_classify_window(self, unclassified_folder, images):
        """Show a Toplevel window for labelling unclassified images one by one.

        Each choice moves the image into the matching training subfolder. The
        keyboard does everything: O/C/T/D choose, Enter takes the model's
        suggestion, U or Ctrl+Z undoes, Escape closes.
        """
        win = self._make_dialog("Review Unclassified Images", min_size=(560, 440))

        state = {
            "index": 0,
            "images": list(images),
            # history entries: (dest_path, src_path) so we can undo by moving back
            "history": [],
            "tk_img": None,     # keep PhotoImage alive
            "tmp_path": None,   # last temp file to clean up
            "suggested": None,  # label the model predicts for the current image
        }
        total = len(state["images"])

        # ── Image ─────────────────────────────────────────────────────────────
        viewer = tk.Frame(win, bg=COLOR_VIEWER_BG)
        viewer.pack(fill="both", expand=True)
        img_label = tk.Label(viewer, bg=COLOR_VIEWER_BG, fg=COLOR_VIEWER_TEXT, font=self.font_bold)
        img_label.pack(expand=True, fill="both", padx=self._px(8), pady=self._px(8))

        # ── Progress and model suggestion ─────────────────────────────────────
        info = ttk.Frame(win, padding=(self._px(14), self._px(10), self._px(14), 0))
        info.pack(fill="x")
        progress_label = ttk.Label(info, style="Heading.TLabel")
        progress_label.pack(side=tk.LEFT)
        name_label = ttk.Label(info, style="Muted.TLabel")
        name_label.pack(side=tk.LEFT, padx=(self._px(10), 0))
        pred_label = tk.Label(info, text="", fg=COLOR_MUTED)
        pred_label.pack(side=tk.RIGHT)

        # ── Buttons ───────────────────────────────────────────────────────────
        bar = ttk.Frame(win, padding=(self._px(14), self._px(10), self._px(14), self._px(4)))
        bar.pack(fill="x")
        choice_buttons = {}
        for label, text, underline in (("open", "Open", 0), ("closed", "Closed", 0),
                                       ("other", "Other", 1), ("discard", "Discard", 0)):
            button = ttk.Button(bar, text=text, underline=underline,
                                command=lambda l=label: classify(l))
            button.pack(side=tk.LEFT, padx=(0, self._px(6)))
            choice_buttons[label] = button

        def _on_classify_close():
            _cleanup_tmp()
            win.destroy()

        ttk.Button(bar, text="Close", command=_on_classify_close).pack(side=tk.RIGHT)
        undo_btn = ttk.Button(bar, text="Undo", underline=0, command=lambda: undo())
        undo_btn.pack(side=tk.RIGHT, padx=(0, self._px(6)))

        ttk.Label(win, style="Muted.TLabel",
                  text="Keys: O open, C closed, T other, D discard. Enter accepts the model's "
                       "suggestion. U undoes. Other and discard move the image to other/ and "
                       "discard/, outside the training set.",
                  wraplength=self._px(640), justify=tk.LEFT).pack(
            fill="x", padx=self._px(14), pady=(0, self._px(12)))

        def _cleanup_tmp():
            p = state.get("tmp_path")
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            state["tmp_path"] = None

        def show_suggestion(label, confidence=None):
            state["suggested"] = label
            for name, button in choice_buttons.items():
                button.configure(default="active" if name == label else "normal")
            if label is None:
                return
            text = f"Model suggests {label.upper()}"
            if confidence is not None:
                text += f" ({confidence:.0%} sure)"
            pred_label.config(text=text, fg=ROOF_STATUS_COLORS.get(label.upper(), COLOR_MUTED))

        def load_current():
            _cleanup_tmp()
            done = state["index"] >= total
            self._set_enabled(undo_btn, bool(state["history"]))
            for button in choice_buttons.values():
                self._set_enabled(button, not done)
            show_suggestion(None)
            if done:
                img_label.config(image="", text="All images reviewed.")
                state["tk_img"] = None
                progress_label.config(text=f"{total} of {total}")
                name_label.config(text="")
                pred_label.config(text="")
                return

            img_name = state["images"][state["index"]]
            img_path = os.path.join(unclassified_folder, img_name)
            progress_label.config(text=f"{state['index'] + 1} of {total}")
            name_label.config(text=self._short_source(img_name, 60))

            img_cv = cv2.imread(img_path)
            if img_cv is None:
                img_label.config(image="", text=f"Could not read {img_name}")
                state["tk_img"] = None
                pred_label.config(text="")
                return

            max_w, max_h = self._px(_CLASSIFY_IMG_MAX_W), self._px(_CLASSIFY_IMG_MAX_H)
            h, w = img_cv.shape[:2]
            scale = min(max_w / w, max_h / h, 1.0)
            disp = cv2.resize(img_cv, (max(1, int(w * scale)), max(1, int(h * scale))))

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(tmp_fd)
            cv2.imwrite(tmp_path, disp)
            state["tmp_path"] = tmp_path

            try:
                tk_img = tk.PhotoImage(file=tmp_path)
                state["tk_img"] = tk_img
                img_label.config(image=tk_img, text="")
            except Exception:
                img_label.config(image="", text=f"Could not display {img_name}")
                state["tk_img"] = None

            # ── Model suggestion ─────────────────────────────────────────────
            if self.model is None:
                pred_label.config(text="Load a model to see its suggestion", fg=COLOR_MUTED)
                return
            try:
                arr = self.prep_image(img_path).flatten().reshape(1, -1)
                prediction = self.model.predict(arr)[0]
                confidence = None
                if callable(getattr(self.model, "predict_proba", None)):
                    confidence = float(max(self.model.predict_proba(arr)[0]))
                show_suggestion("open" if prediction == 1 else "closed", confidence)
            except Exception:
                pred_label.config(text="Model suggestion unavailable", fg=COLOR_MUTED)

        def classify(label):
            if state["index"] >= total:
                return
            img_name = state["images"][state["index"]]
            src_path = os.path.join(unclassified_folder, img_name)

            dest_folder = self._get_training_class_folder(label)
            os.makedirs(dest_folder, exist_ok=True)

            dest_path = os.path.join(dest_folder, img_name)
            base, ext = os.path.splitext(dest_path)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = f"{base}_{counter}{ext}"
                counter += 1

            try:
                shutil.move(src_path, dest_path)
            except Exception as e:
                messagebox.showerror("Could Not Move Image", str(e), parent=win)
                return

            state["history"].append((dest_path, src_path))
            state["index"] += 1
            self.update_training_stats()
            load_current()

        def undo():
            if not state["history"]:
                return
            dest_path, src_path = state["history"].pop()
            try:
                shutil.move(dest_path, src_path)
            except Exception as e:
                messagebox.showerror("Could Not Undo", str(e), parent=win)
                return
            state["index"] = max(0, state["index"] - 1)
            self.update_training_stats()
            load_current()

        def on_key(action):
            def handler(_event):
                action()
                return "break"
            return handler

        for key, label in (("o", "open"), ("c", "closed"), ("t", "other"), ("d", "discard")):
            for sequence in (f"<KeyPress-{key}>", f"<KeyPress-{key.upper()}>"):
                win.bind(sequence, on_key(lambda l=label: classify(l)))
        win.bind("<Return>", on_key(lambda: state["suggested"] and classify(state["suggested"])))
        for sequence in ("<KeyPress-u>", "<KeyPress-U>", "<Control-z>"):
            win.bind(sequence, on_key(undo))
        win.bind("<Escape>", lambda e: _on_classify_close())
        win.protocol("WM_DELETE_WINDOW", _on_classify_close)

        load_current()
        self._center_over_root(win, self._px(_CLASSIFY_IMG_MAX_W + 40),
                               self._px(_CLASSIFY_IMG_MAX_H + 160))
        win.focus_set()

    def convert_fits_to_png(self):
        """Convert FITS images to PNG with debayering and stretching"""
        if not FITS_AVAILABLE:
            messagebox.showerror("FITS Support Missing",
                "Converting FITS files requires astropy:\n\npip install astropy")
            return
            
        # Select FITS files
        fits_files = filedialog.askopenfilenames(
            title="Select FITS files to convert",
            filetypes=[("FITS files", "*.fits"), ("FITS files", "*.fit"), ("All files", "*.*")]
        )
        
        if not fits_files:
            return
            
        # Select output directory
        output_dir = filedialog.askdirectory(title="Select output directory for PNG files")
        if not output_dir:
            return
            
        # Show conversion options dialog
        self.show_fits_conversion_dialog(fits_files, output_dir)

    def show_fits_conversion_dialog(self, fits_files, output_dir):
        """Show dialog for FITS conversion options"""
        dialog = self._make_dialog("Convert FITS to PNG")
        dialog.resizable(False, False)
        dialog.grab_set()
        body = ttk.Frame(dialog, padding=self._px(14))
        body.pack(fill="both", expand=True)

        count = len(fits_files)
        ttk.Label(body, text=f"{count} file{'s' if count != 1 else ''} to {output_dir}",
                  style="Muted.TLabel", wraplength=self._px(380)).pack(anchor="w", pady=(0, self._px(10)))

        # Debayer options
        debayer_frame = self._section(body, "Debayer")
        debayer_frame.pack(fill="x")
        debayer_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(debayer_frame, text="Debayer colour sensor data",
                        variable=debayer_var).pack(anchor="w", pady=(0, self._px(6)))
        debayer_pattern_var = tk.StringVar(value="RGGB")
        pattern_frame = ttk.Frame(debayer_frame)
        pattern_frame.pack(fill="x")
        pattern_widgets = [ttk.Label(pattern_frame, text="Bayer pattern")]
        pattern_widgets[0].pack(side=tk.LEFT, padx=(0, self._px(8)))
        for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
            radio = ttk.Radiobutton(pattern_frame, text=pattern, variable=debayer_pattern_var,
                                    value=pattern)
            radio.pack(side=tk.LEFT, padx=(0, self._px(8)))
            pattern_widgets.append(radio)
        self._enable_with(debayer_var, *pattern_widgets)

        # Stretch options
        stretch_frame = self._section(body, "Stretch")
        stretch_frame.pack(fill="x", pady=(self._px(12), 0))
        stretch_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(stretch_frame, text="Stretch the histogram",
                        variable=stretch_var).pack(anchor="w", pady=(0, self._px(6)))

        stretch_method_var = tk.StringVar(value="percentile")
        method_frame = ttk.Frame(stretch_frame)
        method_frame.pack(fill="x", pady=(0, self._px(6)))
        method_widgets = [ttk.Label(method_frame, text="Method")]
        method_widgets[0].pack(side=tk.LEFT, padx=(0, self._px(8)))
        for text, value in (("Percentile", "percentile"), ("Min–max", "minmax")):
            radio = ttk.Radiobutton(method_frame, text=text, variable=stretch_method_var, value=value)
            radio.pack(side=tk.LEFT, padx=(0, self._px(8)))
            method_widgets.append(radio)

        lower_perc_var = tk.StringVar(value="0.1")
        upper_perc_var = tk.StringVar(value="99.9")
        percentile_frame = ttk.Frame(stretch_frame)
        percentile_frame.pack(fill="x", pady=(0, self._px(6)))
        percentile_widgets = []
        for text, var in (("Low %", lower_perc_var), ("High %", upper_perc_var)):
            label = ttk.Label(percentile_frame, text=text)
            label.pack(side=tk.LEFT)
            entry = ttk.Entry(percentile_frame, textvariable=var, width=7)
            entry.pack(side=tk.LEFT, padx=(self._px(6), self._px(14)))
            percentile_widgets += [label, entry]

        def sync_stretch(*_):
            stretching = stretch_var.get()
            for widget in method_widgets:
                self._set_enabled(widget, stretching)
            for widget in percentile_widgets:
                self._set_enabled(widget, stretching and stretch_method_var.get() == "percentile")

        stretch_var.trace_add("write", sync_stretch)
        stretch_method_var.trace_add("write", sync_stretch)
        sync_stretch()

        # Gamma correction for brightness
        gamma_var = tk.StringVar(value="0.5")
        gamma_frame = ttk.Frame(stretch_frame)
        gamma_frame.pack(fill="x", pady=(0, self._px(6)))
        ttk.Label(gamma_frame, text="Gamma").pack(side=tk.LEFT)
        ttk.Entry(gamma_frame, textvariable=gamma_var, width=7).pack(
            side=tk.LEFT, padx=(self._px(6), self._px(8)))
        ttk.Label(gamma_frame, text="Below 1 brightens, above 1 darkens",
                  style="Muted.TLabel").pack(side=tk.LEFT)

        # Quick presets: (low %, high %, gamma)
        presets = {"Gentle": ("1", "99", "1.0"), "Strong": ("0.1", "99.9", "0.5"),
                   "Very strong": ("0.01", "99.99", "0.3")}
        preset_frame = ttk.Frame(stretch_frame)
        preset_frame.pack(fill="x")
        ttk.Label(preset_frame, text="Presets").pack(side=tk.LEFT, padx=(0, self._px(8)))

        def apply_preset(values):
            lower, upper, gamma = values
            lower_perc_var.set(lower)
            upper_perc_var.set(upper)
            gamma_var.set(gamma)

        for name, values in presets.items():
            ttk.Button(preset_frame, text=name,
                       command=lambda v=values: apply_preset(v)).pack(side=tk.LEFT, padx=(0, self._px(4)))

        # Buttons
        button_frame = ttk.Frame(body)
        button_frame.pack(fill="x", pady=(self._px(14), 0))

        def start_conversion():
            try:
                lower_perc = float(lower_perc_var.get())
                upper_perc = float(upper_perc_var.get())
                gamma = float(gamma_var.get())
                if not (0 <= lower_perc < upper_perc <= 100):
                    raise ValueError("the low percentile must be below the high one, both between 0 and 100")
                if not (0.1 <= gamma <= 5.0):
                    raise ValueError("gamma must be between 0.1 and 5.0")
            except ValueError as e:
                messagebox.showerror("Invalid Settings", f"Check the stretch settings: {e}",
                                     parent=dialog)
                return

            dialog.destroy()
            self.process_fits_conversion(
                fits_files, output_dir,
                debayer_var.get(), debayer_pattern_var.get(),
                stretch_var.get(), stretch_method_var.get(),
                lower_perc, upper_perc, gamma
            )

        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        convert_btn = ttk.Button(button_frame, text="Convert", default="active",
                                 command=start_conversion)
        convert_btn.pack(side=tk.RIGHT, padx=(0, self._px(6)))
        dialog.bind("<Return>", lambda e: start_conversion())
        self._center_over_root(dialog)

    def process_fits_conversion(self, fits_files, output_dir, apply_debayer, debayer_pattern,
                               apply_stretch, stretch_method, lower_perc, upper_perc, gamma=1.0):
        """Process FITS to PNG conversion in a separate thread"""
        def conversion_worker():
            try:
                from astropy.io import fits
                
                total_files = len(fits_files)
                converted = 0
                errors = []
                
                for i, fits_file in enumerate(fits_files):
                    try:
                        # Load FITS file
                        with fits.open(fits_file) as hdul:
                            data = hdul[0].data
                            header = hdul[0].header
                            
                        if data is None:
                            errors.append(f"{os.path.basename(fits_file)}: No image data found")
                            continue
                        
                        # Print diagnostics for debugging
                        print(f"Processing {os.path.basename(fits_file)}:")
                        print(f"  Shape: {data.shape}")
                        print(f"  Data type: {data.dtype}")
                        print(f"  Min: {data.min():.6f}, Max: {data.max():.6f}")
                        print(f"  Mean: {data.mean():.6f}, Std: {data.std():.6f}")
                        
                        # Handle different data types and extreme values
                        if data.dtype in [np.uint8, np.uint16, np.uint32]:
                            # Unsigned integer data
                            data = data.astype(np.float32)
                        elif data.dtype in [np.int8, np.int16, np.int32]:
                            # Signed integer data - may have negative values
                            data = data.astype(np.float32)
                        else:
                            # Already float type
                            data = data.astype(np.float32)
                        
                        # Handle NaN and infinite values
                        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                            print(f"  Warning: Found NaN or infinite values, replacing with median")
                            median_val = np.nanmedian(data)
                            data = np.where(np.isnan(data) | np.isinf(data), median_val, data)
                        
                        # Apply debayering if requested
                        if apply_debayer and len(data.shape) == 2:
                            print(f"  Applying debayering with pattern: {debayer_pattern}")
                            data = self.debayer_image(data, debayer_pattern)
                            print(f"  After debayering: Shape: {data.shape}, Min: {data.min():.6f}, Max: {data.max():.6f}")
                        
                        # Apply stretching if requested
                        if apply_stretch:
                            print(f"  Applying {stretch_method} stretch ({lower_perc}-{upper_perc}%) with gamma={gamma}")
                            data = self.stretch_image(data, stretch_method, lower_perc, upper_perc)
                            print(f"  After stretch: Min: {data.min():.6f}, Max: {data.max():.6f}")
                            
                            # Apply gamma correction for brightness adjustment
                            if gamma != 1.0:
                                print(f"  Applying gamma correction: {gamma}")
                                data = np.power(data, gamma)
                                print(f"  After gamma: Min: {data.min():.6f}, Max: {data.max():.6f}")
                        else:
                            # If no stretching, do basic normalization
                            data_min, data_max = data.min(), data.max()
                            if data_max > data_min:
                                data = (data - data_min) / (data_max - data_min)
                                # Still apply gamma even without stretching
                                if gamma != 1.0:
                                    print(f"  Applying gamma correction: {gamma}")
                                    data = np.power(data, gamma)
                            else:
                                data = np.full_like(data, 0.5)
                            print(f"  Basic normalization with gamma: Min: {data.min():.6f}, Max: {data.max():.6f}")
                        
                        # Final safety check
                        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                            print(f"  Error: Still have NaN/inf after processing, using fallback")
                            data = np.full_like(data, 0.5)
                        
                        # Convert to 8-bit
                        if len(data.shape) == 3:  # Color image
                            data_8bit = (np.clip(data, 0, 1) * 255).astype(np.uint8)
                        else:  # Grayscale
                            data_8bit = (np.clip(data, 0, 1) * 255).astype(np.uint8)
                        
                        print(f"  Final 8-bit: Min: {data_8bit.min()}, Max: {data_8bit.max()}")
                        
                        # Save as PNG
                        base_name = os.path.splitext(os.path.basename(fits_file))[0]
                        output_path = os.path.join(output_dir, f"{base_name}.png")
                        
                        if len(data_8bit.shape) == 3:
                            # Color image - convert RGB to BGR for OpenCV
                            data_bgr = cv2.cvtColor(data_8bit, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(output_path, data_bgr)
                        else:
                            # Grayscale image
                            cv2.imwrite(output_path, data_8bit)
                        
                        converted += 1
                        print(f"Converted {i+1}/{total_files}: {os.path.basename(fits_file)} -> {base_name}.png")
                        self._defer_to_ui(lambda n=i + 1: self._set_activity(
                            f"Converting FITS files: {n} of {total_files}…", clear_after_ms=0))
                        
                    except Exception as e:
                        errors.append(f"{os.path.basename(fits_file)}: {str(e)}")
                        print(f"Error processing {fits_file}: {e}")
                        continue
                
                # Show results. Tk is not thread-safe, so dialogs go via the UI thread.
                message = f"Converted {converted} of {total_files} files to PNG in:\n{output_dir}"
                if errors:
                    message += f"\n\n{len(errors)} failed:\n" + "\n".join(errors[:5])
                    if len(errors) > 5:
                        message += f"\n…and {len(errors) - 5} more"
                self._defer_to_ui(lambda: self._set_activity(
                    f"Converted {converted} of {total_files} FITS files."))
                if errors:
                    # A clean run is reported in the status bar alone
                    self._defer_to_ui(lambda: messagebox.showwarning("Conversion Complete", message))

            except Exception as e:
                self._defer_to_ui(lambda e=e: messagebox.showerror(
                    "Conversion Failed", f"Conversion failed: {e}"))

        self._set_activity(f"Converting FITS files: 0 of {len(fits_files)}…", clear_after_ms=0)
        # Start conversion in background thread
        thread = threading.Thread(target=conversion_worker, daemon=True)
        thread.start()

    def debayer_image(self, data, pattern):
        """Apply debayering to a raw Bayer image"""
        try:
            # Map pattern names to OpenCV constants
            pattern_map = {
                "RGGB": cv2.COLOR_BayerBG2RGB,
                "BGGR": cv2.COLOR_BayerRG2RGB,
                "GRBG": cv2.COLOR_BayerGB2RGB,
                "GBRG": cv2.COLOR_BayerGR2RGB
            }
            
            if pattern not in pattern_map:
                raise ValueError(f"Unknown Bayer pattern: {pattern}")
            
            # Check for valid data range
            data_min, data_max = data.min(), data.max()
            print(f"    Debayer input range: {data_min:.6f} to {data_max:.6f}")
            
            if data_max == data_min:
                print("    Warning: Flat image data, skipping debayering")
                return data
            
            # Normalize data to 0-65535 range for debayering
            # Handle negative values properly
            if data_min < 0:
                # Shift data to positive range first
                data_shifted = data - data_min
                data_norm = (data_shifted / data_shifted.max() * 65535).astype(np.uint16)
            else:
                data_norm = ((data - data_min) / (data_max - data_min) * 65535).astype(np.uint16)
            
            print(f"    Normalized for debayering: {data_norm.min()} to {data_norm.max()}")
            
            # Apply debayering
            debayered = cv2.cvtColor(data_norm, pattern_map[pattern])
            
            # Convert back to float 0-1 range
            result = debayered.astype(np.float32) / 65535.0
            print(f"    Debayered result range: {result.min():.6f} to {result.max():.6f}")
            
            return result
            
        except Exception as e:
            print(f"    Debayering failed: {e}, using original data")
            return data

    def stretch_image(self, data, method, lower_perc, upper_perc):
        """Apply histogram stretching to image data"""
        try:
            if method == "percentile":
                # Use percentile stretching
                if len(data.shape) == 3:  # Color image
                    stretched = np.zeros_like(data)
                    for i in range(data.shape[2]):
                        channel = data[:, :, i]
                        low_val = np.percentile(channel, lower_perc)
                        high_val = np.percentile(channel, upper_perc)
                        
                        # Handle edge case where all values are the same
                        if high_val == low_val:
                            # If all values are the same, just normalize to 0.5
                            stretched[:, :, i] = np.full_like(channel, 0.5)
                        else:
                            stretched[:, :, i] = np.clip((channel - low_val) / (high_val - low_val), 0, 1)
                    return stretched
                else:  # Grayscale
                    low_val = np.percentile(data, lower_perc)
                    high_val = np.percentile(data, upper_perc)
                    
                    # Handle edge case where all values are the same
                    if high_val == low_val:
                        return np.full_like(data, 0.5)
                    else:
                        return np.clip((data - low_val) / (high_val - low_val), 0, 1)
            else:  # min-max stretching
                data_min = data.min()
                data_max = data.max()
                
                # Handle edge case where all values are the same
                if data_max == data_min:
                    return np.full_like(data, 0.5)
                else:
                    return (data - data_min) / (data_max - data_min)
                
        except Exception as e:
            print(f"Stretching failed: {e}, using fallback normalization")
            # Fallback to simple normalization with safety check
            data_min = data.min()
            data_max = data.max()
            if data_max == data_min:
                return np.full_like(data, 0.5)
            else:
                return (data - data_min) / (data_max - data_min)

    def on_log_enabled_changed(self):
        """Called when logging checkbox is toggled"""
        if self.log_enabled.get() and self._log_path_conflicts_with_output():
            self._warn_log_path_conflict("Please set a different path for the classifier log file.")
        self.setup_logging()
        self.save_settings()

    def browse_log_file(self):
        """Browse for log file destination"""
        current_path = self.log_path.get()
        initial_dir = os.path.dirname(current_path) if current_path else os.getcwd()
        path = filedialog.asksaveasfilename(
            title="Select Log File Location",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        if path:
            if os.path.abspath(path) == os.path.abspath(self.output_path.get()):
                messagebox.showerror(
                    "Invalid Log File Path",
                    f"The classifier log file cannot be the same as the roof status output file:\n\n"
                    f"  {self.output_path.get()}\n\n"
                    f"Please choose a different file for the classifier log."
                )
                return
            self.log_path.set(path)
            self.setup_logging()
            self.save_settings()

    def browse_secondary_source(self):
        """Browse for secondary source roof status file"""
        current_path = self.secondary_source_path.get()
        # A URL has no meaningful parent directory to start the dialog in
        if current_path and not self._is_http_source(current_path):
            initial_dir = os.path.dirname(current_path)
        else:
            initial_dir = os.getcwd()
        path = filedialog.askopenfilename(
            title="Select Secondary Roof Status File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        if path:
            self.secondary_source_path.set(path)
            self._secondary_cache = None
            self.save_settings()

    def calculate_next_observation_window(self):
        """Calculate the next safe observation window (when sun is below threshold)"""
        try:
            # Create observer for the given location
            observer = ephem.Observer()
            observer.lat = str(float(self.latitude.get()) * ephem.degree_per_degree)
            observer.lon = str(float(self.longitude.get()) * ephem.degree_per_degree)
            
            # Get threshold angle in radians
            threshold_deg = float(self.sun_angle_threshold.get())
            threshold_rad = threshold_deg * ephem.pi / 180.0
            
            # Create sun object
            sun = ephem.Sun()
            
            # Start from current time
            observer.date = ephem.now()
            current_time = observer.date
            
            # Calculate current sun angle
            sun.compute(observer)
            current_angle = float(sun.alt) * 180.0 / ephem.pi
            
            # If we're already in a safe window, find when it ends
            if current_angle < threshold_deg:
                try:
                    # Find when sun rises above threshold (end of current window)
                    end_time = observer.next_setting(sun, start=current_time)
                    # Actually, we want when it rises above our threshold, not just sets
                    # Let's search forward in small increments
                    search_time = current_time
                    while search_time < current_time + 1:  # Search up to 24 hours ahead
                        observer.date = search_time
                        sun.compute(observer)
                        if float(sun.alt) * 180.0 / ephem.pi > threshold_deg:
                            end_time = search_time
                            break
                        search_time += ephem.minute * 30  # Search in 30-minute increments
                    
                    # Find start of next window (when sun goes below threshold again)
                    search_time = end_time
                    start_next = None
                    while search_time < current_time + 2:  # Search up to 48 hours ahead
                        observer.date = search_time
                        sun.compute(observer)
                        if float(sun.alt) * 180.0 / ephem.pi < threshold_deg:
                            start_next = search_time
                            break
                        search_time += ephem.minute * 30
                    
                    if start_next:
                        # Find end of next window
                        search_time = start_next + ephem.minute * 30
                        end_next = None
                        while search_time < start_next + 1:
                            observer.date = search_time
                            sun.compute(observer)
                            if float(sun.alt) * 180.0 / ephem.pi > threshold_deg:
                                end_next = search_time
                                break
                            search_time += ephem.minute * 30
                        
                        # Return current window end and next window
                        return {
                            'current_end': end_time,
                            'next_start': start_next,
                            'next_end': end_next,
                            'in_window': True
                        }
                    
                except ephem.AlwaysUpError:
                    # Sun never sets (polar summer)
                    return {'error': 'Sun never sets at this location/time'}
                except ephem.NeverUpError:
                    # Sun never rises (polar winter) - always safe
                    return {'always_safe': True}
            else:
                # We're not in a safe window, find when the next one starts
                search_time = current_time
                start_next = None
                while search_time < current_time + 2:  # Search up to 48 hours ahead
                    observer.date = search_time
                    sun.compute(observer)
                    if float(sun.alt) * 180.0 / ephem.pi < threshold_deg:
                        start_next = search_time
                        break
                    search_time += ephem.minute * 30
                
                if start_next:
                    # Find end of next window
                    search_time = start_next + ephem.minute * 30
                    end_next = None
                    while search_time < start_next + 1:
                        observer.date = search_time
                        sun.compute(observer)
                        if float(sun.alt) * 180.0 / ephem.pi > threshold_deg:
                            end_next = search_time
                            break
                        search_time += ephem.minute * 30
                    
                    return {
                        'next_start': start_next,
                        'next_end': end_next,
                        'in_window': False
                    }
                
                return {'error': 'No safe observation window found in next 48 hours'}
                
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error calculating observation window: {e}")
            return {'error': str(e)}

    def calculate_observation_window(self):
        """Calculate the next observation window (when it's safe to report 'open') using UTC"""
        try:
            observer = ephem.Observer()
            observer.lat = str(float(self.latitude.get()))
            observer.lon = str(float(self.longitude.get()))
            threshold = float(self.sun_angle_threshold.get())
            
            sun = ephem.Sun()
            
            # Start checking from current UTC time
            current_time = ephem.now()  # PyEphem uses UTC by default
            observer.date = current_time
            
            # Check current sun position
            sun.compute(observer)
            current_sun_angle = float(sun.alt) * 180.0 / ephem.pi
            
            # If currently safe, find when it becomes unsafe
            if current_sun_angle < threshold:
                # Find when sun rises above threshold (end of current window)
                try:
                    observer.horizon = str(threshold)
                    next_sunrise = observer.next_rising(sun)
                    window_end = next_sunrise
                except:
                    window_end = None  # Never rises above threshold
                
                # Find start of next window (sun sets below threshold)
                try:
                    next_sunset = observer.next_setting(sun)
                    next_window_start = next_sunset
                except:
                    next_window_start = None  # Never sets below threshold
                
                return None, window_end, next_window_start
            
            else:
                # Currently unsafe, find when it becomes safe
                try:
                    observer.horizon = str(threshold)
                    next_sunset = observer.next_setting(sun)
                    window_start = next_sunset
                    
                    # Find end of this window
                    observer.date = next_sunset
                    next_sunrise = observer.next_rising(sun)
                    window_end = next_sunrise
                    
                    return window_start, window_end, None
                    
                except:
                    # Never safe at this location/threshold
                    return "Never", "Never", "Never"
                    
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error calculating observation window: {e}")
            # (None, None, None) used to come back here, which the caller reads as
            # "always safe" - exactly wrong for, say, a mistyped latitude.
            raise

    def format_observation_window(self):
        """Format the observation window for display (times shown in UTC)"""
        try:
            window_start, window_end, next_window_start = self.calculate_observation_window()
            
            def hhmm(value):
                if not value:
                    return "unknown"
                return datetime.strptime(str(value), "%Y/%m/%d %H:%M:%S").strftime("%H:%M")

            if window_start == "Never":
                return "The sun does not get below this limit here today."
            elif window_start is None and window_end is None:
                return "The sun stays below this limit all day."
            elif window_start is None:
                # Currently in safe window
                return (f"Sun is below the limit now, until {hhmm(window_end)} UTC. "
                        f"Next window starts {hhmm(next_window_start)} UTC.")
            else:
                return f"Next window with the sun below the limit: {hhmm(window_start)}–{hhmm(window_end)} UTC"

        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error formatting observation window: {e}")
            return "Can't calculate the observation window. Check the latitude and longitude."

    def update_observation_window_display(self):
        """Refresh the observation window and sun altitude, then again in a minute."""
        if hasattr(self, 'obs_window_label'):
            self.obs_window_label.config(text=self.format_observation_window())
        self._update_sun_status_display()
        # Cancel any pending refresh first, so repeated calls (settings edits,
        # presets) never stack up extra self-perpetuating 60s chains.
        if self._obs_window_after_id is not None:
            try:
                self.root.after_cancel(self._obs_window_after_id)
            except Exception:
                pass
            self._obs_window_after_id = None
        self._obs_window_after_id = self.root.after(60000, self.update_observation_window_display)

    def apply_twilight_preset(self, preset_name):
        """Apply a twilight preset to the sun angle threshold"""
        if preset_name in TWILIGHT_PRESETS:
            self.sun_angle_threshold.set(TWILIGHT_PRESETS[preset_name])
            self.save_settings()
            self.update_observation_window_display()
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Applied twilight preset: {preset_name} ({TWILIGHT_PRESETS[preset_name]}°)")

def _enable_dpi_awareness():
    """Render crisply on scaled Windows displays instead of as a blurry bitmap.

    Must run before the Tk root is created. Tk then scales its fonts to the
    real DPI, and the app scales its own pixel sizes to match (see _px).
    """
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system DPI aware
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def main():
    _enable_dpi_awareness()
    root = tk.Tk()
    RoofClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
