#!/bin/sh

# Add in the 'docs' folder markdown files containing the docstring of the different texthero modules


pydoc-markdown -m preprocessing -I ../texthero > api-preprocessing.md
pydoc-markdown -m representation -I ../texthero > api-representation.md
pydoc-markdown -m visualization -I ../texthero > api-visualization.md
