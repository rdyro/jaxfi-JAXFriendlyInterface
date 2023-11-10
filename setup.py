import os
from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="jaxfi",
    version="0.6.0",
    author="Robert Dyro",
    description="Simplified and user friendly interface to JAX, that behaves like PyTorch.",
    packages=find_packages(),
    long_description=(Path(__file__).absolute().parent / "README.md").read_text(),
)
