# Synthetic RoofStatusFile

**Synthetic RoofStatusFile** is a desktop application for monitoring the state of a telescope's roll-off roof using images from a sky-facing camera. It uses logistic regression to classify each frame as `OPEN` or `CLOSED`, and writes results to a plain text file in the format expected by ASCOM safety monitors and other automation systems.

## How It Works

- You collect a set of example images (`.png`, `.jpg`, or `.jpeg`) of the roof in the open and closed positions.
- You label them using the app (assign each image to "open" or "closed").
- The app trains a lightweight logistic regression model using these examples.
- Once trained, it watches a user-selected folder.
- Every 60 seconds, it checks for the newest image in that folder.
- It classifies the image as `OPEN` or `CLOSED` and overwrites the `.txt` file with the current status as a single line.

The file format looks like this:

```
???2025-06-23 09:39:36AM Roof Status: OPEN
```

## Usage

The app is organized into four tabs: **Training & Model**, **Monitoring**, **Configuration**, and **Utilities**.

### Training & Model

1. Optionally set a **Training Data Folder** to store all training images in one place. If left blank, images are stored in `open/` and `closed/` subfolders relative to the working directory.
2. Use **"Add Frame (Open)"** and **"Add Frame (Closed)"** to import sample images. Duplicate images are detected and skipped automatically.
3. Click **"Train Model"** to fit the classifier. If a model path is already set, the model is saved there automatically.
4. Use **"Load Model"** to load a previously saved `.joblib` file. The last-used model is reloaded automatically on startup.
5. Use **"Save Model As..."** to save the current model to a new location.
6. Use **"Validate Model"** to check accuracy against a folder of labelled images. The folder must contain `open/` and `closed/` subfolders.
7. Use **"Benchmark Models"** to compare multiple `.joblib` files side by side against the fixed validation set.
8. Set a **Fixed Validation Set Folder** to use the same test set each time you validate or benchmark without being prompted.

To build up the training set over time, enable **"Save random samples while monitoring"** and set a sample rate (for example, `0.1` saves roughly 10% of checked frames to an `unclassified/` subfolder). Click **"Classify Images"** to work through those images and move each one to `open/`, `closed/`, `other/`, or discard it.

### Monitoring

1. Set the **Monitor Folder** (the folder where your camera saves images).
2. Set the **Output Status File** path (`RoofStatusFile.txt` is the default).
3. Click **"Start Monitoring"** to begin. The status bar at the bottom of the window shows the current state and provides a quick toggle button.
4. The app checks the newest image every 60 seconds and appends a line to the output file.

### Configuration

**Logging** — Enable file logging and choose a log file path. The log records each classification decision, sun angle, and secondary source status.

**Observatory Location** — Enter your latitude and longitude (decimal degrees). These are used to calculate the sun's current elevation angle.

**Sun Angle Threshold** — If the sun is above this angle (in degrees), any `OPEN` classification is overridden to `CLOSED`. This prevents the output file from reporting the roof open during daytime even if the image classifier disagrees. Preset buttons are provided for common twilight definitions:

- Sunset / Sunrise (0.0 degrees)
- Civil (-6.0 degrees)
- Nautical (-12.0 degrees)
- Astronomical (-18.0 degrees)

The calculated observation window (next sunset and sunrise in UTC) is displayed and updated automatically.

**Secondary Roof Status File** — Enable this option and point it at another `RoofStatusFile.txt`-format file to cross-reference a second source. The secondary status is shown in the monitoring display and logged alongside each primary classification, but it does not override the primary decision.

**ASCOM Alpaca Safety Monitor** — The app can serve an ASCOM Alpaca-compatible Safety Monitor API on a configurable port (default 11111). Enable it here and configure the port and device number. N.I.N.A. and other ASCOM clients can auto-discover the device via UDP port 32227 or connect manually. Use **"Test Discovery"** to verify the network setup, and **"Open Setup Page"** to view the Alpaca setup endpoint in a browser.

### Utilities

**Convert FITS to PNG** — Batch-convert FITS images to PNG with optional debayering (RGGB, BGGR, GRBG, GBRG) and histogram stretching (percentile or min-max). This is useful for preparing training images captured directly from an astronomy camera.

## Installation

### Download Pre-built Executable

The easiest way to get started is to download the latest pre-built executable from the [Releases page](https://github.com/bortleorg/Synthetic_RoofStatusFile.txt/releases).

1. Go to the [Releases page](https://github.com/bortleorg/Synthetic_RoofStatusFile.txt/releases)
2. Download the latest `Synthetic_RoofStatusFile.exe`
3. Run the executable directly - no installation required.

### Building from Source

If you prefer to build from source or need to modify the code:

```bash
pip install -r requirements.txt
pip install pyinstaller
pyinstaller synthetic_roofstatus.spec
```

The built executable will be in the `dist/` folder.

## System Requirements

- Windows 10 or later
- Python 3.11+ (only needed for building from source)
- Minimal CPU usage when idle

## Notes

- This is not a deep learning system. It uses classical ML for speed and simplicity.
- Images should be reasonably consistent in angle and framing.
- Works best when lighting or exposure is fairly stable across captures.
- Settings (model path, folder paths, location, thresholds, etc.) are saved automatically to `roof_classifier_settings.json` in the working directory and restored on the next launch.
