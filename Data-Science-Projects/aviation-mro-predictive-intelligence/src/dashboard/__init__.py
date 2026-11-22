"""Dashboard package for the Aviation MRO Predictive Intelligence Platform.

This package contains a Streamlit application that visualises the saved
analytical artefacts produced by Sections 9.1-9.6. It never fabricates data:
every number shown is read from a JSON report, a Parquet dataset, or a saved
model card produced by the analytics pipelines.
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
