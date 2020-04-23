import setuptools
from texthero.version import Version
from setuptools import find_packages

import os

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

# If building on RTD, don't install anything
if os.environ.get("READTHEDOCS", None) == "True":
    install_requires = []

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="texthero",
    version=Version("1.0.2").number,
    description="Text preprocessing, representation and visualization made easy.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Jonathan Besomi",
    author_email="jonathanbesomi@gmail.com",
    url="https://github.com/jbesomi/texthero",
    keywords = ['text analytics'],
    install_requires=['nltk', 'scikit-learn', 'plotly_express'],
    license="MIT",
    zip_safe=False,
    packages=find_packages(),
    classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3'
  ]
)
