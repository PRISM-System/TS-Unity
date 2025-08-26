import importlib
from typing import Optional


def _load_module(module_name: str) -> Optional[object]:
    """Try to import a model submodule from forecasting or anomaly_detection.

    Returns the imported module object or None if not found.
    """
    for subpkg in ("forecasting", "anomaly_detection"):
        try:
            return importlib.import_module(f"{__name__}.{subpkg}.{module_name}")
        except Exception:
            continue
    return None


# Forecasting model modules commonly used by experiments
_model_modules = [
    "Autoformer", "Transformer", "TimesNet", "Nonstationary_Transformer", "DLinear",
    "FEDformer", "Informer", "LightTS", "Reformer", "ETSformer", "Pyraformer",
    "PatchTST", "MICN", "Crossformer", "FiLM", "iTransformer", "Koopa", "TiDE",
    "FreTS", "MambaSimple", "TemporalFusionTransformer", "SCINet", "PAttn", "TimeXer",
    "WPMixer", "MultiPatchFormer",
]

# Anomaly detection models (exposed for completeness)
_anomaly_modules = [
    "AnomalyTransformer", "OmniAnomaly", "USAD", "DAGMM", "LSTM_AE", "LSTM_VAE", "VTTPAT", "VTTSAT",
]

for _name in _model_modules + _anomaly_modules:
    _mod = _load_module(_name)
    if _mod is not None:
        globals()[_name] = _mod

del importlib, Optional, _load_module, _model_modules, _anomaly_modules, _name, _mod

