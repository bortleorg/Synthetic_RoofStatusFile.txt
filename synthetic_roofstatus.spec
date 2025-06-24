# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/synthetic_roofstatus.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'ephem',
        'pytz',
        'astropy',
        'astropy.io',
        'astropy.io.fits',
        'scipy',
        'scipy.sparse',
        'scipy.sparse.csgraph',
        'scipy.sparse._matrix',
        'scipy.sparse._sparsetools',
        'scipy.sparse._csparsetools',
        'scipy._lib',
        'scipy._lib._ccallback_c',
        'scipy.special',
        'scipy.special._ufuncs_cxx',
        'sklearn',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
        'sklearn.ensemble._gradient_boosting',
        'sklearn.linear_model',
        'sklearn.linear_model._logistic',
        'sklearn.metrics',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Collect all scipy and sklearn submodules - but be more selective to avoid issues
from PyInstaller.utils.hooks import collect_all

try:
    scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')
    a.datas += scipy_datas
    a.binaries += scipy_binaries
    a.hiddenimports += scipy_hiddenimports
except Exception as e:
    print(f"Warning: Could not collect scipy modules: {e}")

try:
    sklearn_datas, sklearn_binaries, sklearn_hiddenimports = collect_all('sklearn')
    a.datas += sklearn_datas
    a.binaries += sklearn_binaries
    a.hiddenimports += sklearn_hiddenimports
except Exception as e:
    print(f"Warning: Could not collect sklearn modules: {e}")

try:
    astropy_datas, astropy_binaries, astropy_hiddenimports = collect_all('astropy')
    a.datas += astropy_datas
    a.binaries += astropy_binaries
    a.hiddenimports += astropy_hiddenimports
except Exception as e:
    print(f"Warning: Could not collect astropy modules: {e}")

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
