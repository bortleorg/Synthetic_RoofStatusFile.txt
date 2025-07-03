# PyInstaller hook for sklearn
# This ensures that all necessary sklearn compiled extensions are included

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

# Collect sklearn submodules that are commonly missed
hiddenimports = [
    'sklearn.utils._cython_blas',
    'sklearn.utils._seq_dataset', 
    'sklearn.utils._random',
    'sklearn.utils._weight_vector',
    'sklearn.utils._logistic_sigmoid',
    'sklearn.neighbors._typedefs',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree._utils',
    'sklearn.tree._tree',
    'sklearn.tree._splitter',
    'sklearn.tree._criterion',
    'sklearn.ensemble._gradient_boosting',
    'sklearn.linear_model._cd_fast',
    'sklearn.linear_model._sgd_fast',
    'sklearn.linear_model._sag_fast',
    'sklearn.metrics._pairwise_distances_reduction',
]

# Collect all sklearn submodules
hiddenimports += collect_submodules('sklearn.linear_model')
hiddenimports += collect_submodules('sklearn.preprocessing')
hiddenimports += collect_submodules('sklearn.metrics')
hiddenimports += collect_submodules('sklearn.utils')
hiddenimports += collect_submodules('sklearn.base')

# Collect data files and binaries
datas = collect_data_files('sklearn')
binaries = collect_dynamic_libs('sklearn')
