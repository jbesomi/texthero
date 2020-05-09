#!/bin/sh

cd ..

echo "Formatting code ..."
./format.sh


echo "Checking code ..."
./check.sh

echo "Updating doc ..."
cd website/docs
./create_docs_with_sphinx.sh

cd -

python3 setup.py sdist bdist_wheel
twine upload dist/*
