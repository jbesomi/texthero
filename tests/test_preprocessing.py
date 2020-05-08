from texthero import preprocessing
import pandas as pd
"""
Test `remove_digits`
"""

# Check block 
s = pd.Series("remove block of digits 1234 h1n1")
s_true = pd.Series("remove block of digits   h1n1")
assert preprocessing.remove_digits(s).equals(s_true)

# Check with only_blocks = False
s = pd.Series("remove block of digits 1234 h1n1")
s_true = pd.Series("remove block of digits   h n ")
assert preprocessing.remove_digits(s, only_blocks=False).equals(s_true)

# Check in brackets
s = pd.Series("Digits in bracket (123 $) needs to be cleaned out")
s_true = pd.Series("Digits in bracket (  $) needs to be cleaned out")
assert preprocessing.remove_digits(s).equals(s_true)

# Check start digits
s = pd.Series("123 starting digits needs to be cleaned out")
s_true = pd.Series("  starting digits needs to be cleaned out")
assert preprocessing.remove_digits(s).equals(s_true)

# Check end digits
s = pd.Series("end digits needs to be cleaned out 123")
s_true = pd.Series("end digits needs to be cleaned out  ")
assert preprocessing.remove_digits(s).equals(s_true)

# Check with punctuation
s = pd.Series("check.with.123.punctuation!?+")
s_true = pd.Series("check.with. .punctuation!?+")
assert preprocessing.remove_digits(s).equals(s_true)


"""
Test `remove_punctuations`
"""

text = "hello."
text_preprocessed = "hello "

assert preprocessing.remove_punctuation(pd.Series(text)).equals(
    pd.Series(text_preprocessed))
"""
Test `remove_diacritics`
"""

text = "hèllo"
text_preprocessed = "hello"

assert preprocessing.remove_diacritics(pd.Series(text)).equals(
    pd.Series(text_preprocessed))
"""
Test `remove whitespaces`
"""

text = "hello   world  hello        world "
text_preprocessed = "hello world hello world"

assert preprocessing.remove_whitespace(pd.Series(text)).equals(
    pd.Series(text_preprocessed))
