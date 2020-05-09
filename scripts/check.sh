#!/bin/sh

# if any command inside script returns error, exit and return that error
set -e 

# ensure that we're always inside the root of our application
cd "${0%/*}/.."

echo "Test doctest."
cd tests
./test_doctest.sh
cd ..

python3 -m unittest discover -s tests -t .

#npm run build

echo "Failed! --" && exit 1
