#!/bin/sh


echo "Test doctest."
cd ../tests
./test_doctest.sh
cd ..

python3 -m unittest discover -s tests -t .

#npm run build
