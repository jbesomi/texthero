#!/bin/sh

# if any command inside script returns error, exit and return that error
set -e 

# ensure that we're always inside the root of our application
cd "${0%/*}/.."

cd scripts

echo "Format code."
./format.sh


echo "Update documentation."
./update_documentation.sh


echo "Test code."
./tests.sh

#cd website
#npm run build
