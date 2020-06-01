#!/bin/sh

cd ../docs
make html
./to_docusaurus.py
