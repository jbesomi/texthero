#!/bin/sh

# Run check.
./scripts/check.sh

# $? stores exit value of the last command
if [ $? -ne 0 ]; then
 echo "All tests must pass before commit."
 exit 1
fi
