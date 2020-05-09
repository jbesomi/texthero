#!/bin/sh

# if any command inside script returns error, exit and return that error
set -e 

# ensure that we're always inside the root of our application
cd "${0%/*}/.."

echo "Formatting code."
cd scripts
./format.sh
cd ..


echo "Test doctest."
cd tests
./test_doctest.sh
cd ..


echo "Unittest."
python3 -m unittest discover -s tests -t .

cd website
npm run build
