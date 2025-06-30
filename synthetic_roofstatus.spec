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
        # Scipy compiled extensions that are often missed
        'scipy._lib._ccallback_c',
        'scipy._cyutility',
        'scipy.sparse._sparsetools',
        'scipy.sparse._csparsetools',
        'scipy.sparse.csgraph._validation',
        'scipy.special._ufuncs_cxx',
        'scipy.special._ellip_harm_2',
        # Additional sklearn dependencies
        'sklearn.utils._cython_blas',
        'sklearn.utils._seq_dataset',
        'sklearn.utils._random',
        'sklearn.utils._weight_vector',
        'sklearn.neighbors._typedefs',
        'sklearn.neighbors._quad_tree',
        'sklearn.tree._utils',
    ],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude problematic optional dependencies
        'torch',
        'matplotlib',
        'scipy._lib.array_api_compat.torch',
        'sklearn.externals.array_api_compat.torch',
        'astropy.visualization.wcsaxes',
        # Exclude other heavy dependencies we don't need
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'plotly',
        'bokeh',
        'seaborn',
        'sympy',
        'PIL.ImageQt',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'tkinter.test',
        # Exclude test modules
        'sklearn.tests',
        'scipy.tests',
        'numpy.tests',
        'astropy.tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Let the hooks handle most of the scipy/sklearn collection
# Only add specific modules that might still be missed
try:
    print("Adding additional hidden imports...")
    additional_imports = [
        # Core scipy modules that are sometimes missed
        'scipy._cyutility',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        # Core sklearn modules
        'sklearn.utils._typedefs',
        'sklearn.metrics._dist_metrics',
    ]
    a.hiddenimports.extend(additional_imports)
    print("Successfully added additional hidden imports")
except Exception as e:
    print(f"Warning: Could not add additional imports: {e}")

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
