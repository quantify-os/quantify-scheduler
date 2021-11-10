#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("AUTHORS.rst") as authors_file:
    authors = authors_file.read()

with open("CHANGELOG.rst") as history_file:
    history = history_file.read()

with open("requirements.txt") as installation_requirements_file:
    requirements = installation_requirements_file.read().splitlines()

with open("requirements_setup.txt") as setup_requirements_file:
    setup_requirements = setup_requirements_file.read().splitlines()

with open("requirements_dev.txt") as test_requirements_file:
    test_requirements = test_requirements_file.read().splitlines()

version = "0.5.1"

setup(
    author="The Quantify consortium consisting of Qblox and Orange Quantum Systems",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Quantify-scheduler is a python package for writing quantum programs "
    "featuring a hybrid gate-pulse control model with explicit timing control.",
    install_requires=requirements,
    license="BSD-4 license",
    long_description=readme + "\n\n" + authors + "\n\n" + history,
    include_package_data=True,
    keywords="quantify-scheduler",
    name="quantify-scheduler",
    packages=find_packages(include=["quantify_scheduler", "quantify_scheduler.*"]),
    package_data={"": ["*.json"]},  # ensures JSON schema are included
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://gitlab.com/quantify-os/quantify-scheduler",
    version=version,
    zip_safe=False,
)
