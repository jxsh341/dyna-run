from setuptools import setup, find_packages

setup(
    name="dyna-run",
    version="0.1.0",
    description="Dynamic Sparse AI Inference Runtime",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0",
        "pandas>=2.0.0",
        "plotly>=5.14.0",
        "matplotlib>=3.7.0",
        "streamlit>=1.28.0",
        "requests>=2.31.0",
    ],
)
