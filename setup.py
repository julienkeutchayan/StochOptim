# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stochoptim",
    version="0.0.5",
    author="Julien Keutchayan",
    author_email="j.keutchayan@gmail.com",
    description=("StochOptim is a Stochastic Optimization package with scenario generation tools "
                 "for two- and multi-stage problems"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/julienkeutchayan/StochOptim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)