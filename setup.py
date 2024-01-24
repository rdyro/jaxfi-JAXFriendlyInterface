from pathlib import Path
from setuptools import setup, find_packages
import toml

config = toml.loads((Path(__file__).parent / "pyproject.toml").read_text())

setup(
    name="jaxfi",
    version=config["project"]["version"],
    author="Robert Dyro",
    description="Simplified and user friendly interface to JAX, that behaves like PyTorch.",
    packages=find_packages(),
    long_description=(Path(__file__).absolute().parent / "README.md").read_text(),
)