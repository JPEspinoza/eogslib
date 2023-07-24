"""
This file compiles the eOGS library
"""

from setuptools import setup
from Cython.Build import cythonize
import os

os.environ["CC"] = "gcc-12"
os.environ["CXX"] = "gcc-12"

setup(
    name="eogslib",
    ext_modules=cythonize(
        "eogslib.pyx",
        language_level="3"
    ),
)