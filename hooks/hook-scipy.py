# PyInstaller hook for scipy
# This ensures that all necessary scipy compiled extensions are included

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

# Collect all scipy submodules
hiddenimports = collect_submodules('scipy')

# Collect data files (needed for some scipy functionality)
datas = collect_data_files('scipy')

# Collect dynamic libraries (compiled extensions)
binaries = collect_dynamic_libs('scipy')

# Specifically include known problematic modules
hiddenimports += [
    'scipy._cyutility',
    'scipy._lib._ccallback_c',
    'scipy.sparse._sparsetools', 
    'scipy.sparse._csparsetools',
    'scipy.sparse.csgraph._validation',
    'scipy.special._ufuncs_cxx',
    'scipy.special._ellip_harm_2',
    'scipy.linalg._fblas',
    'scipy.linalg._flapack',
    'scipy.linalg._decomp_update',
    'scipy.linalg._solve_toeplitz',
    'scipy.optimize._lbfgsb',
    'scipy.optimize._moduleTNC',
    'scipy.optimize._cobyla',
    'scipy.optimize._slsqp',
    'scipy.optimize._minpack',
    'scipy.optimize._lsq',
    'scipy.optimize._zeros',
    'scipy.interpolate._fitpack',
    'scipy.interpolate._bsplines',
    'scipy.ndimage._nd_image',
    'scipy.spatial._distance_wrap',
    'scipy.spatial._voronoi',
    'scipy.spatial._qhull',
    'scipy.io.matlab._mio_utils',
    'scipy.io.matlab._mio5_utils',
]
