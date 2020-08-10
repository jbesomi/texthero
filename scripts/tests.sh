#!/bin/sh

cd ..

python3 -m unittest discover -b -s tests -t .
