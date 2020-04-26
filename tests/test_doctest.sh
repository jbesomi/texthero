#!/bin/sh

cd ../texthero/

python preprocessing.py
python representation.py
python visualization.py
