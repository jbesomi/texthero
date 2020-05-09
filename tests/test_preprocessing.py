from texthero import preprocessing
import pandas as pd
import string
"""
Test remove_digits
"""

# Check block 
s = pd.Series("remove block of digits 1234 h1n1")
s_true = pd.Series("remove block of digits  h1n1")
assert preprocessing.remove_digits(s).equals(s_true)

# Check with only_blocks = False
s = pd.Series("remove block of digits 1234 h1n1")
s_true = pd.Series("remove block of digits  hn")
assert preprocessing.remove_digits(s, only_blocks=False).equals(s_true)

# Check in brackets
s = pd.Series("Digits in bracket (123 $) needs to be cleaned out")
s_true = pd.Series("Digits in bracket ( $) needs to be cleaned out")
assert preprocessing.remove_digits(s).equals(s_true)

# Check start digits
s = pd.Series("123 starting digits needs to be cleaned out")
s_true = pd.Series(" starting digits needs to be cleaned out")
assert preprocessing.remove_digits(s).equals(s_true)

# Check end digits
s = pd.Series("end digits needs to be cleaned out 123")
s_true = pd.Series("end digits needs to be cleaned out ")
assert preprocessing.remove_digits(s).equals(s_true)

# Check with punctuation
s = pd.Series("1.2.3")
s_true = pd.Series("..")
assert preprocessing.remove_digits(s).equals(s_true)

# Check two consecutive numbers
s = pd.Series("+41 1234 5678")
s_true = pd.Series("+  ")
assert preprocessing.remove_digits(s).equals(s_true)

# Check other symbol sytays.
s = pd.Series(string.punctuation)
s_true = pd.Series(string.punctuation)
assert preprocessing.remove_digits(s).equals(s_true)

"""
Test remove_punctuations
"""

s = pd.Series("hello.")
s_true = pd.Series("hello ")
assert preprocessing.remove_punctuation(s).equals(s_true)


"""
Test remove_diacritics
"""

s = pd.Series("hèllo")
s_true = pd.Series("hello")
assert preprocessing.remove_diacritics(s).equals(s_true)

"""
Test remove whitespaces
"""

s = pd.Series("hello   world  hello        world ")
s_true = pd.Series("hello world hello world")
assert preprocessing.remove_whitespace(s).equals(s_true)

"""
Text previous bugs.
"""


s = pd.Series("E-I-E-I-O\nAnd on")
s_true = pd.Series("e-i-e-i-o\n ")
pipeline = [preprocessing.lowercase, preprocessing.remove_stopwords]
assert preprocessing.clean(s, pipeline=pipeline).equals(s_true)
