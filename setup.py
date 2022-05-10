import os
from setuptools import setup, find_packages

requirementPath = "requirements.txt"

with open(requirementPath) as f:
    install_requires = f.read().splitlines()

setup(
    name = 'ultrasound',
    packages = find_packages(),
    install_requires=install_requires
)
