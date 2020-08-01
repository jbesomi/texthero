#!/bin/sh

cd ..

coverage run -m unittest discover -s tests -t .

coverage report -m
coverage html
