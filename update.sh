#!/bin/sh

twine upload dist/*
python3 setup.py sdist bdist_wheel
