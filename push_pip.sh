#!/bin/sh

echo "Formatting code ..."
./format.sh


echo "Checking code ..."
./check.sh

python3 setup.py sdist bdist_wheel
twine upload dist/*
