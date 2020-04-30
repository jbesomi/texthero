from texthero import preprocessing
import pandas as pd
"""
Test `remove_digits`
"""

text = "remove_digits remove all the 1234 digits of a pandas series. H1N1"
text_preprocessed = "remove_digits remove all the digits of a pandas series. H1N1"
text_preprocessed_block_false = "remove_digits remove all the  digits of a pandas series. HN"
assert preprocessing.remove_digits(pd.Series(text)).equals(
    pd.Series(text_preprocessed))

assert preprocessing.remove_digits(
    pd.Series(text), only_blocks=False).equals(
        pd.Series(text_preprocessed_block_false))
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
Test `removepaces`
"""

text = "hello   world  hello        world "
text_preprocessed = "hello world hello world"

assert preprocessing.remove_whitespace(pd.Series(text)).equals(
    pd.Series(text_preprocessed))
