# Synthetic RoofStatusFile

**Synthetic RoofStatusFile** is a minimal desktop application for monitoring the state of a telescope's roll-off roof using images from a sky-facing camera. It uses basic machine learning to classify each frame as `OPEN` or `CLOSED`, and writes results to a plain text file in the format expected by ASCOM saferty monitor and other automation systems.

## How It Works

- You collect a small set of example images (`.png`) of the roof in the **open** and **closed** positions.
- You label them using the app (assign each image to "open" or "shut").
- The app trains a lightweight logistic regression model using these examples.
- Once trained, it watches a user-selected folder.
- Every 60 seconds, it checks for the newest `.png` image in that folder.
- It classifies the image as `OPEN` or `CLOSED` and appends a log entry to a `.txt` file.

The log format looks like this:

```
2025-06-23 09:39:36PM Roof Status: OPEN
2025-06-23 11:07:47AM Roof Status: CLOSED
```

## Usage

1. Launch the application.
2. Use **"Add Frame (Open)"** and **"Add Frame (Shut)"** to collect sample images.
3. Click **"Train Model"** to create a classifier. Save it to a `.joblib` file if you want to reuse it.
4. Load a previously saved model if available using **"Load Model"**.
5. Set the **Monitor Folder** (where your camera drops images).
6. Set the **Output Status File** path (`RoofStatusFile.txt` is the default).
7. Click **"Start Monitoring"** to begin automatic classification.
8. The app will classify a new image every 60 seconds and append to the log file.

## Installation

### Download Pre-built Executable

The easiest way to get started is to download the latest pre-built executable from the [Releases page](https://github.com/bortleorg/Synthetic_RoofStatusFile.txt/releases).

1. Go to the [Releases page](https://github.com/bortleorg/Synthetic_RoofStatusFile.txt/releases)
2. Download the latest `Synthetic_RoofStatusFile.exe`
3. Run the executable directly - no installation required!

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
- Minimal CPU usage (<1% when idle, fast classification)

## Notes

- This is not a deep learning system. It uses classical ML for speed and simplicity.
- Images should be reasonably consistent in angle and framing.
- Works best when lighting or exposure is fairly stable across captures.
