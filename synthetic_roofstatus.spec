# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/synthetic_roofstatus.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Core dependencies - only include what we actually use
        'ephem',
        'pytz',
        'astropy.io.fits',
        'sklearn.linear_model._logistic',
        'sklearn.preprocessing._data',
        'sklearn.metrics._classification',
        'sklearn.utils._weight_vector',
        'sklearn.utils._random',
        'joblib',
        'cv2',
        'numpy',
        # TKinter related
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.ttk',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude problematic optional dependencies
        'torch',
        'matplotlib',
        'scipy._lib.array_api_compat.torch',
        'sklearn.externals.array_api_compat.torch',
        'astropy.visualization.wcsaxes',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Only collect essential sklearn modules to avoid dependency issues
try:
    from PyInstaller.utils.hooks import collect_submodules
    # Only get the specific sklearn modules we need
    sklearn_modules = [
        'sklearn.linear_model',
        'sklearn.preprocessing', 
        'sklearn.metrics',
        'sklearn.utils',
        'sklearn.base'
    ]
    for module in sklearn_modules:
        try:
            submodules = collect_submodules(module)
            a.hiddenimports.extend(submodules)
            print(f"Collected {module}")
        except Exception as e:
            print(f"Warning: Could not collect {module}: {e}")
except Exception as e:
    print(f"Warning: Could not collect sklearn modules: {e}")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='Synthetic_RoofStatusFile',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='Synthetic_RoofStatusFile',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
