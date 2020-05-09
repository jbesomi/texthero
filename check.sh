cd tests
./test_doctest.sh

python3 test_preprocessing.py
python3 test_representation.py
python3 test_nlp.py

npm run build
