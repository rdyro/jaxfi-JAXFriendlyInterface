import os
from setuptools import setup

# borrowed from https://pythonhosted.org/an_example_pypi_project/setuptools.html


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="jfi",
    version="0.3.0",
    author="Robert Dyro",
    description=("Simplified and user friendly interface to JAX."),
    license="MIT",
    packages=["jfi"],
    long_description=read("README.md"),
)
