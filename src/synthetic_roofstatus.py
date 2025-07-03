import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import cv2
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import threading
import time
from datetime import datetime
import hashlib
import json
import logging
import shutil
import ephem
import pytz
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
        
        self.model = None
        self.stop_monitor = False
        self.logger = None
        self.ascom_server = None
        self.load_settings()
        self.setup_logging()
        self.setup_gui()

    def load_settings(self):
        """Load settings from JSON file"""
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)
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
                'ascom_device_number': self.ascom_device_number.get()
            }
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def setup_logging(self):
        """Setup logging configuration"""
        if hasattr(self, 'logger') and self.logger:
            # Remove existing handlers
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
        
        self.logger = logging.getLogger('RoofClassifier')
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        if self.log_enabled.get():
            try:
                # File handler
                file_handler = logging.FileHandler(self.log_path.get())
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Error setting up file logging: {e}")
        
        # Console handler (always enabled)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def start_ascom_server(self):
        """Start the ASCOM Alpaca server"""
        if not FLASK_AVAILABLE:
            messagebox.showerror("Error", "Flask is not installed. Please install flask and flask-cors to use ASCOM Alpaca functionality.")
            return
            
        if self.ascom_server:
            messagebox.showwarning("Warning", "ASCOM server is already running.")
            return
            
        try:
            port = int(self.ascom_port.get())
            device_number = int(self.ascom_device_number.get())
            
            self.ascom_server = AscomAlpacaSafetyMonitor(
                port=port,
                device_number=device_number,
                roof_classifier_app=self
            )
            
            # Start server in a separate thread
            server_thread = threading.Thread(
                target=self.ascom_server.run,
                daemon=True
            )
            server_thread.start()
            
            if self.logger:
                self.logger.info(f"ASCOM Alpaca server started on port {port}")
            
            messagebox.showinfo("ASCOM Server Started", 
                f"ASCOM Alpaca Safety Monitor started on port {port}\n"
                f"Device number: {device_number}\n"
                f"Management API: http://localhost:{port}/management/apiversions\n"
                f"Configure NINA to connect to: localhost:{port}")
                
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for port and device number.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start ASCOM server: {str(e)}")
            
    def stop_ascom_server(self):
        """Stop the ASCOM Alpaca server"""
        if self.ascom_server:
            try:
                self.ascom_server.stop()
                self.ascom_server = None
                if self.logger:
                    self.logger.info("ASCOM Alpaca server stopped")
                messagebox.showinfo("ASCOM Server Stopped", "ASCOM Alpaca server has been stopped.")
            except Exception as e:
                messagebox.showerror("Error", f"Error stopping ASCOM server: {str(e)}")
        else:
            messagebox.showwarning("Warning", "ASCOM server is not running.")
            
    def on_ascom_enabled_changed(self):
        """Called when ASCOM enabled checkbox is toggled"""
        if self.ascom_enabled.get():
            self.start_ascom_server()
        else:
            self.stop_ascom_server()
        self.save_settings()
        
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

    def calculate_sun_angle(self):
        """Calculate the sun's elevation angle for the given location using UTC"""
        try:
            # Create observer for the given location
            observer = ephem.Observer()
            observer.lat = str(float(self.latitude.get()))
            observer.lon = str(float(self.longitude.get()))
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
            return 0.0  # Default to 0 if calculation fails

    def is_sun_safe_for_open(self):
        """Check if sun angle is safe to report 'open' status"""
        try:
            sun_angle = self.calculate_sun_angle()
            threshold = float(self.sun_angle_threshold.get())
            is_safe = sun_angle < threshold
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Sun angle: {sun_angle:.1f}°, Threshold: {threshold:.1f}°, Safe for open: {is_safe}")
            
            return is_safe
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error checking sun safety: {e}")
            return True  # Default to safe if calculation fails

    def read_secondary_source(self):
        """Read the secondary source roof status file"""
        if not self.secondary_source_enabled.get() or not self.secondary_source_path.get():
            return None, None
        
        try:
            file_path = self.secondary_source_path.get()
            if not os.path.exists(file_path):
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning(f"Secondary source file not found: {file_path}")
                return None, None
            
            # Get file modification time in UTC
            mod_time = datetime.utcfromtimestamp(os.path.getmtime(file_path))
            
            # Read the last line of the file
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.warning(f"Secondary source file is empty: {file_path}")
                    return None, None
                
                last_line = lines[-1].strip()
                
                # Try to parse the status from the line
                if "OPEN" in last_line.upper():
                    status = "OPEN"
                elif "CLOSED" in last_line.upper():
                    status = "CLOSED"
                else:
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

    def setup_gui(self):
        # Training section
        train_frame = tk.LabelFrame(self.root, text="Training Data", padx=5, pady=5)
        train_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(train_frame, text="Add Frame (Open)", command=lambda: self.add_frame("open")).pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Add Frame (Closed)", command=lambda: self.add_frame("closed")).pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Clear Training Data", command=self.clear_training_data).pack(side=tk.LEFT, padx=5)
        
        # Stats display
        self.stats_label = tk.Label(train_frame, text="Training set: Open: 0, Closed: 0", fg="blue")
        self.stats_label.pack(pady=5)
        self.update_training_stats()

        # Model section
        model_frame = tk.LabelFrame(self.root, text="Model", padx=5, pady=5)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(model_frame, text="Train Model", command=self.train_model).pack(side=tk.LEFT, padx=5)
        tk.Button(model_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        tk.Button(model_frame, text="Validate Model", command=self.validate_model).pack(side=tk.LEFT, padx=5)
        tk.Button(model_frame, text="Save Model As...", command=self.save_current_model_as).pack(side=tk.LEFT, padx=5)
        
        # Model path with browse button
        model_path_frame = tk.Frame(model_frame)
        model_path_frame.pack(fill="x", pady=5)
        tk.Label(model_path_frame, text="Default Save Path for New Models:").pack(anchor="w")
        path_entry_frame = tk.Frame(model_path_frame)
        path_entry_frame.pack(fill="x")
        tk.Entry(path_entry_frame, textvariable=self.model_path, width=40).pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(path_entry_frame, text="Browse...", command=self.browse_model_path).pack(side=tk.RIGHT, padx=(5,0))

        # Monitoring section
        monitor_frame = tk.LabelFrame(self.root, text="Monitoring", padx=5, pady=5)
        monitor_frame.pack(fill="x", padx=10, pady=5)
        
        # Monitor folder with browse button
        monitor_folder_frame = tk.Frame(monitor_frame)
        monitor_folder_frame.pack(fill="x", pady=2)
        tk.Label(monitor_folder_frame, text="Monitor Folder:").pack(anchor="w")
        folder_entry_frame = tk.Frame(monitor_folder_frame)
        folder_entry_frame.pack(fill="x")
        tk.Entry(folder_entry_frame, textvariable=self.monitor_path, width=40).pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(folder_entry_frame, text="Browse...", command=self.browse_monitor_folder).pack(side=tk.RIGHT, padx=(5,0))
        
        # Output file with browse button
        output_file_frame = tk.Frame(monitor_frame)
        output_file_frame.pack(fill="x", pady=2)
        tk.Label(output_file_frame, text="Output Status File:").pack(anchor="w")
        output_entry_frame = tk.Frame(output_file_frame)
        output_entry_frame.pack(fill="x")
        tk.Entry(output_entry_frame, textvariable=self.output_path, width=40).pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(output_entry_frame, text="Browse...", command=self.browse_output_file).pack(side=tk.RIGHT, padx=(5,0))
        
        button_frame = tk.Frame(monitor_frame)
        button_frame.pack(pady=5)
        tk.Button(button_frame, text="Start Monitoring", command=self.start_monitoring).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Stop Monitoring", command=self.stop_monitoring).pack(side=tk.LEFT, padx=5)

        # Monitoring status display
        status_frame = tk.Frame(monitor_frame)
        status_frame.pack(fill="x", pady=5)
        self.status_label = tk.Label(status_frame, text="Monitoring: Not active", fg="gray")
        self.status_label.pack()
        self.countdown_label = tk.Label(status_frame, text="", fg="blue")
        self.countdown_label.pack()

        # Configuration section
        config_frame = tk.LabelFrame(self.root, text="Configuration", padx=5, pady=5)
        config_frame.pack(fill="x", padx=10, pady=5)
        
        # Logging configuration
        log_frame = tk.Frame(config_frame)
        log_frame.pack(fill="x", pady=2)
        
        log_checkbox = tk.Checkbutton(log_frame, text="Enable Logging to File", 
                                     variable=self.log_enabled, command=self.on_log_enabled_changed)
        log_checkbox.pack(side=tk.LEFT)
        
        log_path_frame = tk.Frame(log_frame)
        log_path_frame.pack(side=tk.RIGHT, fill="x", expand=True, padx=(10,0))
        tk.Entry(log_path_frame, textvariable=self.log_path, width=30).pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(log_path_frame, text="Browse...", command=self.browse_log_file).pack(side=tk.RIGHT, padx=(5,0))
        
        # Observatory location
        location_frame = tk.Frame(config_frame)
        location_frame.pack(fill="x", pady=2)
        tk.Label(location_frame, text="Observatory Location:").pack(side=tk.LEFT)
        tk.Label(location_frame, text="Lat:").pack(side=tk.LEFT, padx=(10,0))
        tk.Entry(location_frame, textvariable=self.latitude, width=8).pack(side=tk.LEFT, padx=(2,5))
        tk.Label(location_frame, text="Lon:").pack(side=tk.LEFT)
        tk.Entry(location_frame, textvariable=self.longitude, width=8).pack(side=tk.LEFT, padx=(2,5))
        
        # Twilight threshold configuration
        twilight_frame = tk.Frame(config_frame)
        twilight_frame.pack(fill="x", pady=2)
        tk.Label(twilight_frame, text="Sun Angle Threshold:").pack(side=tk.LEFT)
        
        # Manual threshold entry
        tk.Entry(twilight_frame, textvariable=self.sun_angle_threshold, width=6).pack(side=tk.LEFT, padx=(2,0))
        tk.Label(twilight_frame, text="°").pack(side=tk.LEFT)
        
        # Twilight presets
        preset_frame = tk.Frame(config_frame)
        preset_frame.pack(fill="x", pady=2)
        tk.Label(preset_frame, text="Presets:").pack(side=tk.LEFT)
        for preset_name in TWILIGHT_PRESETS.keys():
            tk.Button(preset_frame, text=preset_name, 
                     command=lambda p=preset_name: self.apply_twilight_preset(p)).pack(side=tk.LEFT, padx=2)
        

        # Observation window display
        window_frame = tk.Frame(config_frame)
        window_frame.pack(fill="x", pady=5)
        self.obs_window_label = tk.Label(window_frame, text="Calculating observation window...", 
                                   fg="darkgreen", justify=tk.LEFT, font=("Arial", 9))
        self.obs_window_label.pack(side=tk.LEFT)
        tk.Button(window_frame, text="Refresh", command=self.update_observation_window_display).pack(side=tk.RIGHT)
        
        # UTC time note
        utc_note_frame = tk.Frame(config_frame)
        utc_note_frame.pack(fill="x", pady=2)
        tk.Label(utc_note_frame, text="All times UTC", 
                fg="gray", font=("Arial", 8)).pack(anchor="w")
        
        # Update window display after GUI is set up
        self.root.after(1000, self.update_observation_window_display)

        # Secondary source
        secondary_frame = tk.Frame(config_frame)
        secondary_frame.pack(fill="x", pady=2)
        
        secondary_checkbox = tk.Checkbutton(secondary_frame, text="Monitor Secondary Roof Status File", 
                                          variable=self.secondary_source_enabled)
        secondary_checkbox.pack(side=tk.LEFT)
        
        secondary_path_frame = tk.Frame(secondary_frame)
        secondary_path_frame.pack(side=tk.RIGHT, fill="x", expand=True, padx=(10,0))
        tk.Entry(secondary_path_frame, textvariable=self.secondary_source_path, width=30).pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(secondary_path_frame, text="Browse...", command=self.browse_secondary_source).pack(side=tk.RIGHT, padx=(5,0))

        # ASCOM Alpaca configuration section
        if FLASK_AVAILABLE:
            ascom_frame = tk.LabelFrame(self.root, text="ASCOM Alpaca Safety Monitor", padx=5, pady=5)
            ascom_frame.pack(fill="x", padx=10, pady=5)
            
            # Enable ASCOM checkbox
            ascom_enable_frame = tk.Frame(ascom_frame)
            ascom_enable_frame.pack(fill="x", pady=2)
            
            ascom_checkbox = tk.Checkbutton(ascom_enable_frame, text="Enable ASCOM Alpaca Safety Monitor", 
                                          variable=self.ascom_enabled, command=self.on_ascom_enabled_changed)
            ascom_checkbox.pack(side=tk.LEFT)
            
            # Port and device number configuration
            ascom_config_frame = tk.Frame(ascom_frame)
            ascom_config_frame.pack(fill="x", pady=2)
            
            tk.Label(ascom_config_frame, text="Port:").pack(side=tk.LEFT)
            tk.Entry(ascom_config_frame, textvariable=self.ascom_port, width=6).pack(side=tk.LEFT, padx=(2,10))
            
            tk.Label(ascom_config_frame, text="Device Number:").pack(side=tk.LEFT)
            tk.Entry(ascom_config_frame, textvariable=self.ascom_device_number, width=6).pack(side=tk.LEFT, padx=(2,10))
            
            # Manual control buttons
            ascom_buttons_frame = tk.Frame(ascom_frame)
            ascom_buttons_frame.pack(fill="x", pady=2)
            
            tk.Button(ascom_buttons_frame, text="Start ASCOM Server", 
                     command=self.start_ascom_server).pack(side=tk.LEFT, padx=5)
            tk.Button(ascom_buttons_frame, text="Stop ASCOM Server", 
                     command=self.stop_ascom_server).pack(side=tk.LEFT, padx=5)
            tk.Button(ascom_buttons_frame, text="Test Discovery", 
                     command=self.test_ascom_discovery).pack(side=tk.LEFT, padx=5)
            tk.Button(ascom_buttons_frame, text="Open Setup Page", 
                     command=self.open_ascom_setup_page).pack(side=tk.LEFT, padx=5)
            
            # Information
            ascom_info_frame = tk.Frame(ascom_frame)
            ascom_info_frame.pack(fill="x", pady=2)
            
            info_text = ("Configure NINA to connect to this Safety Monitor:\n"
                        "• NINA will auto-discover this device (recommended)\n"
                        "• Or manually configure:\n"
                        "  - Device Type: Safety Monitor (Alpaca)\n"
                        "  - IP Address: localhost or your computer's IP\n"
                        "  - Port: (as configured above)\n"
                        "  - Device Number: (as configured above)\n"
                        "• Discovery runs on UDP port 32227\n"
                        "• Use 'Test Discovery' to verify network setup")
            
            tk.Label(ascom_info_frame, text=info_text, font=("Arial", 8), 
                    fg="darkgreen", justify=tk.LEFT).pack(anchor="w")
        else:
            # Show message if Flask is not available
            flask_frame = tk.LabelFrame(self.root, text="ASCOM Alpaca Safety Monitor", padx=5, pady=5)
            flask_frame.pack(fill="x", padx=10, pady=5)
            
            tk.Label(flask_frame, 
                    text="ASCOM Alpaca functionality requires Flask and Flask-CORS.\n"
                         "Run: pip install flask flask-cors",
                    fg="red", justify=tk.LEFT).pack(anchor="w")

        # Utilities section
        utils_frame = tk.LabelFrame(self.root, text="Utilities", padx=5, pady=5)
        utils_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(utils_frame, text="Convert FITS to PNG", command=self.convert_fits_to_png).pack(side=tk.LEFT, padx=5)

    def get_image_hash(self, image_path):
        """Generate a hash of the image content to detect duplicates"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return hashlib.md5(img.tobytes()).hexdigest()

    def get_existing_hashes(self, label):
        """Get hashes of all existing images in the training folder"""
        hashes = set()
        if os.path.isdir(label):
            for file in os.listdir(label):
                if file.lower().endswith(".png"):
                    try:
                        hash_val = self.get_image_hash(os.path.join(label, file))
                        hashes.add(hash_val)
                    except:
                        continue
        return hashes

    def add_frame(self, label):
        files = filedialog.askopenfilenames(filetypes=[("PNG files", "*.png")])
        if not files:
            return
        os.makedirs(label, exist_ok=True)
        
        existing_hashes = self.get_existing_hashes(label)
        count = 0
        duplicates = 0
        
        for path in files:
            # Check if this image is already in the training set
            try:
                img_hash = self.get_image_hash(path)
                if img_hash in existing_hashes:
                    duplicates += 1
                    continue
                    
                dest = os.path.join(label, os.path.basename(path))
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
                
        message = f"{count} {label} frame(s) saved."
        if duplicates > 0:
            message += f" {duplicates} duplicate(s) skipped."
        messagebox.showinfo("Saved", message)
        self.update_training_stats()

    def update_training_stats(self):
        """Update the training statistics display"""
        open_count = 0
        closed_count = 0
        
        if os.path.isdir("open"):
            open_count = len([f for f in os.listdir("open") if f.lower().endswith(".png")])
        if os.path.isdir("closed"):
            closed_count = len([f for f in os.listdir("closed") if f.lower().endswith(".png")])
            
        self.stats_label.config(text=f"Training set: Open: {open_count}, Closed: {closed_count}")

    def clear_training_data(self):
        """Clear all training data"""
        result = messagebox.askyesno("Confirm", "Are you sure you want to delete all training data?")
        if not result:
            return
            
        import shutil
        for folder in ["open", "closed"]:
            if os.path.isdir(folder):
                shutil.rmtree(folder)
        messagebox.showinfo("Cleared", "All training data has been cleared.")
        self.update_training_stats()

    def train_model(self):
        X, y = [], []
        for label, val in [("open", 1), ("closed", 0)]:
            if not os.path.isdir(label):
                continue
            for file in os.listdir(label):
                if file.lower().endswith(".png"):
                    img = self.prep_image(os.path.join(label, file))
                    X.append(img.flatten())
                    y.append(val)
        if not X:
            messagebox.showerror("Error", "No training data found.")
            return
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        
        # Show training summary
        open_count = sum(1 for label in y if label == 1)
        closed_count = sum(1 for label in y if label == 0)
        message = f"Model trained successfully!\nTraining samples: Open: {open_count}, Closed: {closed_count}"
        
        if self.model_path.get():
            dump(clf, self.model_path.get())
            message += f"\nModel saved to {self.model_path.get()}"
            self.save_settings()
        else:
            message += "\n(Model not saved - specify a path to save)"
        messagebox.showinfo("Model Trained", message)
        self.model = clf

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Joblib model", "*.joblib")])
        if path:
            self.model = load(path)
            self.model_path.set(path)
            self.save_settings()
            messagebox.showinfo("Loaded", f"Loaded model from {path}")

    def save_model_as(self):
        current_path = self.model_path.get()
        initial_dir = os.path.dirname(current_path) if current_path else os.getcwd()
        path = filedialog.asksaveasfilename(
            defaultextension=".joblib", 
            filetypes=[("Joblib model", "*.joblib")],
            initialdir=initial_dir
        )
        if path:
            self.model_path.set(path)
            self.save_settings()

    def prep_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img

    def validate_model(self):
        """Run validation on a separate set of test images"""
        if not self.model:
            messagebox.showerror("Error", "Load or train a model first.")
            return
            
        # Let user select validation folder
        folder = filedialog.askdirectory(title="Select folder with validation images")
        if not folder:
            return
            
        # Look for 'open' and 'closed' subfolders
        open_folder = os.path.join(folder, "open")
        closed_folder = os.path.join(folder, "closed")
        
        if not os.path.isdir(open_folder) and not os.path.isdir(closed_folder):
            messagebox.showerror("Error", "Validation folder must contain 'open' and/or 'closed' subfolders.")
            return
            
        X_val, y_val, file_names = [], [], []
        
        # Load validation data
        for label, val in [("open", 1), ("closed", 0)]:
            val_folder = os.path.join(folder, label)
            if not os.path.isdir(val_folder):
                continue
            for file in os.listdir(val_folder):
                if file.lower().endswith(".png"):
                    try:
                        img = self.prep_image(os.path.join(val_folder, file))
                        X_val.append(img.flatten())
                        y_val.append(val)
                        file_names.append(f"{label}/{file}")
                    except Exception as e:
                        print(f"Error processing {file}: {e}")
                        
        if not X_val:
            messagebox.showerror("Error", "No validation images found.")
            return
            
        # Make predictions
        X_val = np.array(X_val)
        y_pred = self.model.predict(X_val)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        
        # Create detailed results window
        self.show_validation_results(y_val, y_pred, file_names, accuracy)

    def show_validation_results(self, y_true, y_pred, file_names, accuracy):
        """Display validation results in a new window"""
        results_window = tk.Toplevel(self.root)
        results_window.title("Validation Results")
        results_window.geometry("600x400")
        
        # Summary
        summary_frame = tk.Frame(results_window)
        summary_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(summary_frame, text=f"Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)", 
                font=("Arial", 12, "bold")).pack()
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tk.Label(summary_frame, text=f"Confusion Matrix:").pack()
        tk.Label(summary_frame, text=f"True Closed/Predicted Closed: {cm[0,0]}, True Closed/Predicted Open: {cm[0,1]}").pack()
        tk.Label(summary_frame, text=f"True Open/Predicted Closed: {cm[1,0]}, True Open/Predicted Open: {cm[1,1]}").pack()
        
        # Detailed results
        tk.Label(results_window, text="Detailed Results:").pack()
        
        # Create scrollable text widget
        text_frame = tk.Frame(results_window)
        text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Show individual predictions
        for i, (true_label, pred_label, filename) in enumerate(zip(y_true, y_pred, file_names)):
            true_str = "OPEN" if true_label == 1 else "CLOSED"
            pred_str = "OPEN" if pred_label == 1 else "CLOSED"
            correct = "✓" if true_label == pred_label else "✗"
            text_widget.insert(tk.END, f"{correct} {filename}: True={true_str}, Predicted={pred_str}\n")
            
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        text_widget.config(state=tk.DISABLED)

    def classify_latest_png(self):
        folder = self.monitor_path.get()
        if not self.model or not os.path.isdir(folder):
            if self.logger:
                self.logger.error("No model loaded or invalid monitor folder")
            return None, "No model or invalid folder"
        
        pngs = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
        if not pngs:
            if self.logger:
                self.logger.warning("No PNG files found in monitor folder")
            return None, "No PNG files found"
        
        latest = max(pngs, key=lambda f: os.path.getmtime(os.path.join(folder, f)))
        img_path = os.path.join(folder, latest)
        
        # Get secondary source status for comparison
        secondary_status, secondary_time = self.read_secondary_source()
        
        # Classify the image
        img = self.prep_image(img_path).flatten().reshape(1, -1)
        pred = self.model.predict(img)[0]
        image_status = "OPEN" if pred == 1 else "CLOSED"
        
        # Apply sun angle guard rails
        sun_safe = self.is_sun_safe_for_open()
        if image_status == "OPEN" and not sun_safe:
            if self.logger:
                self.logger.warning(f"Image classification suggests OPEN, but sun angle too high - overriding to CLOSED")
            final_status = "CLOSED"
            override_reason = " (Sun too high - safety override)"
        else:
            final_status = image_status
            override_reason = ""
        
        # Log the analysis
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        sun_angle = self.calculate_sun_angle()
        
        log_message = f"Image: {latest}, Raw prediction: {image_status}, Final status: {final_status}"
        log_message += f", Sun angle: {sun_angle:.1f}°"
        
        if secondary_status:
            log_message += f", Secondary source: {secondary_status} (updated: {secondary_time})"
        else:
            log_message += ", Secondary source: Not available"
        
        if self.logger:
            self.logger.info(log_message)
        
        # Write to output file
        line = f"{now} Roof Status: {final_status}{override_reason}\n"
        with open(self.output_path.get(), "a") as f:
            f.write(line)
        
        print(f"[{final_status}] {latest}")
        return latest, final_status

    def update_monitoring_status(self, filename, status):
        """Update the monitoring status display"""
        if filename and status:
            # Get secondary source info for display
            secondary_status, secondary_time = self.read_secondary_source()
            
            status_text = f"Last checked: {filename} → {status}"
            if secondary_status:
                time_str = secondary_time.strftime("%H:%M:%S UTC") if secondary_time else "Unknown"
                status_text += f" | Secondary: {secondary_status} ({time_str})"
            else:
                status_text += " | Secondary: N/A"
            
            self.status_label.config(text=status_text, fg="green")
        else:
            self.status_label.config(text="Monitoring: Error checking files", fg="red")

    def update_countdown(self, seconds_remaining):
        """Update the countdown display"""
        if seconds_remaining > 0:
            self.countdown_label.config(text=f"Next check in: {seconds_remaining}s")
            
            # Update observation window display every 5 minutes (300 seconds)
            if seconds_remaining % 300 == 0:
                self.update_observation_window_display()
        else:
            self.countdown_label.config(text="Checking now...")
            # Update observation window when checking
            self.update_observation_window_display()

    def clear_monitoring_status(self):
        """Clear the monitoring status when stopped"""
        self.status_label.config(text="Monitoring: Stopped", fg="gray")
        self.countdown_label.config(text="")

    def monitor_loop(self):
        check_interval = 60  # 60 seconds between checks
        
        while not self.stop_monitor:
            # Check the latest PNG and update status
            filename, status = self.classify_latest_png()
            self.root.after(0, lambda f=filename, s=status: self.update_monitoring_status(f, s))
            
            # Countdown loop with 1-second updates
            for remaining in range(check_interval, 0, -1):
                if self.stop_monitor:
                    break
                self.root.after(0, lambda r=remaining: self.update_countdown(r))
                time.sleep(1)
            
        # Clear status when monitoring stops
        self.root.after(0, self.clear_monitoring_status)

    def start_monitoring(self):
        if not self.model:
            messagebox.showerror("Error", "Load or train a model first.")
            return
        
        # Validate configuration
        try:
            float(self.latitude.get())
            float(self.longitude.get())
            float(self.sun_angle_threshold.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for latitude, longitude, and sun angle threshold.")
            return
        
        # Setup logging with current settings
        self.setup_logging()
        
        self.stop_monitor = False
        self.status_label.config(text="Monitoring: Starting...", fg="blue")
        self.countdown_label.config(text="")
        
        if self.logger:
            self.logger.info("Monitoring started")
            self.logger.info(f"Observatory location: {self.latitude.get()}°, {self.longitude.get()}°")
            self.logger.info(f"Sun angle threshold: {self.sun_angle_threshold.get()}°")
            if self.secondary_source_enabled.get():
                self.logger.info(f"Secondary source enabled: {self.secondary_source_path.get()}")
            else:
                self.logger.info("Secondary source disabled")
        
        threading.Thread(target=self.monitor_loop, daemon=True).start()
        self.update_observation_window_display()  # Start periodic updates
        messagebox.showinfo("Started", "Monitoring started.")

    def stop_monitoring(self):
        self.stop_monitor = True
        if self.logger:
            self.logger.info("Monitoring stopped")
        messagebox.showinfo("Stopped", "Monitoring stopped.")

    def browse_model_path(self):
        """Browse for default model save location for new trained models"""
        current_path = self.model_path.get()
        initial_dir = os.path.dirname(current_path) if current_path else os.getcwd()
        path = filedialog.asksaveasfilename(
            title="Select Default Save Location for New Models",
            defaultextension=".joblib",
            filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        if path:
            self.model_path.set(path)
            self.save_settings()

    def save_current_model_as(self):
        """Save the currently loaded model to a new location"""
        if not self.model:
            messagebox.showerror("Error", "No model is currently loaded. Train or load a model first.")
            return
            
        current_path = self.model_path.get()
        initial_dir = os.path.dirname(current_path) if current_path else os.getcwd()
        path = filedialog.asksaveasfilename(
            title="Save Current Model As...",
            defaultextension=".joblib", 
            filetypes=[("Joblib model", "*.joblib")],
            initialdir=initial_dir
        )
        if path:
            try:
                dump(self.model, path)
                messagebox.showinfo("Saved", f"Current model saved to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {e}")

    def save_model_as(self):
        """Legacy method - redirects to save_current_model_as for compatibility"""
        self.save_current_model_as()

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

    def convert_fits_to_png(self):
        """Convert FITS images to PNG with debayering and stretching"""
        if not FITS_AVAILABLE:
            messagebox.showerror("Error", 
                "FITS support not available. Please install astropy:\n"
                "pip install astropy")
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
        dialog = tk.Toplevel(self.root)
        dialog.title("FITS to PNG Conversion Options")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Debayer options
        debayer_frame = tk.LabelFrame(dialog, text="Debayer Options", padx=5, pady=5)
        debayer_frame.pack(fill="x", padx=10, pady=5)
        
        debayer_var = tk.BooleanVar(value=True)
        tk.Checkbutton(debayer_frame, text="Apply debayering", variable=debayer_var).pack(anchor="w")
        
        debayer_pattern_var = tk.StringVar(value="RGGB")
        tk.Label(debayer_frame, text="Bayer pattern:").pack(anchor="w")
        pattern_frame = tk.Frame(debayer_frame)
        pattern_frame.pack(fill="x")
        for pattern in ["RGGB", "BGGR", "GRBG", "GBRG"]:
            tk.Radiobutton(pattern_frame, text=pattern, variable=debayer_pattern_var, 
                          value=pattern).pack(side=tk.LEFT)
        
        # Stretch options
        stretch_frame = tk.LabelFrame(dialog, text="Stretch Options", padx=5, pady=5)
        stretch_frame.pack(fill="x", padx=10, pady=5)
        
        stretch_var = tk.BooleanVar(value=True)
        tk.Checkbutton(stretch_frame, text="Apply histogram stretch", variable=stretch_var).pack(anchor="w")
        
        tk.Label(stretch_frame, text="Stretch method:").pack(anchor="w")
        stretch_method_var = tk.StringVar(value="percentile")
        method_frame = tk.Frame(stretch_frame)
        method_frame.pack(fill="x")
        tk.Radiobutton(method_frame, text="Percentile", variable=stretch_method_var, 
                      value="percentile").pack(side=tk.LEFT)
        tk.Radiobutton(method_frame, text="Min-Max", variable=stretch_method_var, 
                      value="minmax").pack(side=tk.LEFT)
        
        # Percentile options
        percentile_frame = tk.Frame(stretch_frame)
        percentile_frame.pack(fill="x", pady=2)
        tk.Label(percentile_frame, text="Lower percentile:").pack(side=tk.LEFT)
        lower_perc_var = tk.StringVar(value="0.1")
        tk.Entry(percentile_frame, textvariable=lower_perc_var, width=5).pack(side=tk.LEFT, padx=5)
        tk.Label(percentile_frame, text="Upper percentile:").pack(side=tk.LEFT)
        upper_perc_var = tk.StringVar(value="99.9")
        tk.Entry(percentile_frame, textvariable=upper_perc_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Gamma correction for brightness
        gamma_frame = tk.Frame(stretch_frame)
        gamma_frame.pack(fill="x", pady=2)
        tk.Label(gamma_frame, text="Gamma correction (0.1=bright, 1.0=normal, 3.0=dark):").pack(anchor="w")
        gamma_entry_frame = tk.Frame(gamma_frame)
        gamma_entry_frame.pack(fill="x")
        tk.Label(gamma_entry_frame, text="Gamma:").pack(side=tk.LEFT)
        gamma_var = tk.StringVar(value="0.5")
        tk.Entry(gamma_entry_frame, textvariable=gamma_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Quick preset buttons
        preset_frame = tk.Frame(stretch_frame)
        preset_frame.pack(fill="x", pady=2)
        tk.Label(preset_frame, text="Quick presets:").pack(anchor="w")
        preset_buttons_frame = tk.Frame(preset_frame)
        preset_buttons_frame.pack(fill="x")
        
        def apply_conservative():
            lower_perc_var.set("1")
            upper_perc_var.set("99")
            gamma_var.set("1.0")
        
        def apply_aggressive():
            lower_perc_var.set("0.1")
            upper_perc_var.set("99.9")
            gamma_var.set("0.5")
        
        def apply_very_aggressive():
            lower_perc_var.set("0.01")
            upper_perc_var.set("99.99")
            gamma_var.set("0.3")
        
        tk.Button(preset_buttons_frame, text="Conservative", command=apply_conservative).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_buttons_frame, text="Aggressive", command=apply_aggressive).pack(side=tk.LEFT, padx=2)
        tk.Button(preset_buttons_frame, text="Very Aggressive", command=apply_very_aggressive).pack(side=tk.LEFT, padx=2)
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def start_conversion():
            try:
                lower_perc = float(lower_perc_var.get())
                upper_perc = float(upper_perc_var.get())
                gamma = float(gamma_var.get())
                if not (0 <= lower_perc < upper_perc <= 100):
                    raise ValueError("Invalid percentile range")
                if not (0.1 <= gamma <= 5.0):
                    raise ValueError("Gamma must be between 0.1 and 5.0")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid values: {e}")
                return
                
            dialog.destroy()
            self.process_fits_conversion(
                fits_files, output_dir,
                debayer_var.get(), debayer_pattern_var.get(),
                stretch_var.get(), stretch_method_var.get(),
                lower_perc, upper_perc, gamma
            )
        
        tk.Button(button_frame, text="Convert", command=start_conversion).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

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
                        
                    except Exception as e:
                        errors.append(f"{os.path.basename(fits_file)}: {str(e)}")
                        print(f"Error processing {fits_file}: {e}")
                        continue
                
                # Show results
                message = f"Conversion complete!\n{converted}/{total_files} files converted successfully."
                if errors:
                    message += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])
                    if len(errors) > 5:
                        message += f"\n... and {len(errors)-5} more errors"
                
                messagebox.showinfo("Conversion Complete", message)
                
            except Exception as e:
                messagebox.showerror("Conversion Error", f"Conversion failed: {str(e)}")
        
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
            self.log_path.set(path)
            self.setup_logging()
            self.save_settings()

    def browse_secondary_source(self):
        """Browse for secondary source roof status file"""
        current_path = self.secondary_source_path.get()
        initial_dir = os.path.dirname(current_path) if current_path else os.getcwd()
        path = filedialog.askopenfilename(
            title="Select Secondary Roof Status File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        if path:
            self.secondary_source_path.set(path)
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
            return None, None, None

    def format_observation_window(self):
        """Format the observation window for display (times shown in UTC)"""
        try:
            window_start, window_end, next_window_start = self.calculate_observation_window()
            
            if window_start == "Never":
                return "Observation Window: Never safe at this location/time"
            elif window_start is None and window_end is None:
                return "Observation Window: Always safe at this location/time"
            elif window_start is None:
                # Currently in safe window
                end_str = datetime.strptime(str(window_end), "%Y/%m/%d %H:%M:%S").strftime("%H:%M UTC") if window_end else "Unknown"
                next_str = datetime.strptime(str(next_window_start), "%Y/%m/%d %H:%M:%S").strftime("%H:%M UTC") if next_window_start else "Unknown"
                return f"Current Window: Now → {end_str} | Next: {next_str} → ..."
            else:
                # Waiting for next window
                start_str = datetime.strptime(str(window_start), "%Y/%m/%d %H:%M:%S").strftime("%H:%M UTC") if window_start else "Unknown"
                end_str = datetime.strptime(str(window_end), "%Y/%m/%d %H:%M:%S").strftime("%H:%M UTC") if window_end else "Unknown"
                return f"Next Window: {start_str} → {end_str}"
                
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error formatting observation window: {e}")
            return "Observation Window: Error calculating"

    def update_observation_window_display(self):
        """Update the observation window display"""
        if hasattr(self, 'obs_window_label'):
            self.obs_window_label.config(text=self.format_observation_window())
        # Schedule next update in 60 seconds if monitoring
        if not self.stop_monitor:
            self.root.after(60000, self.update_observation_window_display)

    def apply_twilight_preset(self, preset_name):
        """Apply a twilight preset to the sun angle threshold"""
        if preset_name in TWILIGHT_PRESETS:
            self.sun_angle_threshold.set(TWILIGHT_PRESETS[preset_name])
            self.save_settings()
            self.update_observation_window_display()
            if hasattr(self, 'logger') and self.logger:
                self.logger.info(f"Applied twilight preset: {preset_name} ({TWILIGHT_PRESETS[preset_name]}°)")

if __name__ == "__main__":
    root = tk.Tk()
    app = RoofClassifierApp(root)
    root.mainloop()
