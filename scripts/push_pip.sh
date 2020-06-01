#!/bin/sh

echo "Formatting code ..."
./format.sh

echo "Checking code ..."
./check.sh

echo "Updating doc ..."
cd ../docs/
make html
./to_docusaurus.py
cd ..

python3 setup.py sdist bdist_wheel
twine upload --skip-existing dist/*
