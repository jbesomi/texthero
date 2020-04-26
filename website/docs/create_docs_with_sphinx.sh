#!/bin/sh

cd sphinx
sphinx-build -M markdown ./ ../temp_md_output

cd ../temp_md_output

# do NOT copy index.md
cp ./markdown/api-preprocessing.md ../
cp ./markdown/api-representation.md ../
cp ./markdown/api-visualization.md ../

cd ..
# delete folder
rm -rfd temp_md_output

# Add 'docusaurus' id to files
echo "---\nid: api-preprocessing \ntitle: Preprocessing\n---\n\n$(cat api-preprocessing.md)" > api-preprocessing.md
echo "---\nid: api-representation \ntitle: Representation\n---\n\n$(cat api-representation.md)" > api-representation.md
echo "---\nid: api-visualization \ntitle: Visualization\n---\n\n$(cat api-visualization.md)" > api-visualization.md
