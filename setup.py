import os
from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="jfi",
    version="0.4.0",
    author="Robert Dyro",
    description=("Simplified and user friendly interface to JAX."),
    license="MIT",
    packages=find_packages(),
    long_description=(Path(__file__).absolute().parent / "README.md").read_text(),
)
