"""Setup script for xl_mimo_beam_training package."""

from setuptools import setup, find_packages

setup(
    name="xl_mimo_beam_training",
    version="1.0.0",
    description="Near-Field Beam Training for XL-MIMO Using Deep Learning",
    author="J. Nie, Y. Cui et al.",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
