#!/bin/sh

cd ..

yapf texthero --recursive -i
yapf tests --recursive -i
