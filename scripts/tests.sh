#!/bin/sh

cd ..

python3 -m unittest discover -s tests -t .
