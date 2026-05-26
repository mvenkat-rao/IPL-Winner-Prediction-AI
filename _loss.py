"""
Compatibility shim: some pickled models reference a top-level module named
`_loss` (from older or differently-built sklearn). This module re-exports
`sklearn._loss` so unpickling can find the expected symbols.
"""
try:
    from sklearn._loss import *  # re-export symbols
except Exception as e:
    # Provide a helpful error if sklearn._loss isn't available
    raise ImportError("sklearn._loss is required for unpickling legacy models: " + str(e))
