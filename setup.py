import texthero
import setuptools
from setuptools import find_packages

import os

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=texthero.__about__.__name__,
    version=texthero.__about__.__version__,
    description=texthero.__about__.__description__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=texthero.__about__.__author__,
    url=texthero.__about__.__url__,
    keywords = ['text analytics'],
    install_requires=['nltk', 'scikit-learn', 'plotly_express'],
    license=texthero.__about__.__license__,
    zip_safe=False,
    packages=find_packages(),
    classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3'
  ]
)
