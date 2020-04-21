from texthero import preprocessing
import pandas as pd

"""
Test `remove_digits_s`
"""

text = "remove_digits_s remove all the 1234 digits of a pandas series. H1N1"
text_preprocessed = "remove_digits_s remove all the digits of a pandas series. H1N1"
text_preprocessed_block_false = "remove_digits_s remove all the   digits of a pandas series. H N "
assert preprocessing.remove_digits_s(pd.Series([text])).equals(
    pd.Series([text_preprocessed])
)

assert preprocessing.remove_digits_s(pd.Series([text]), only_blocks=False).equals(
    pd.Series([text_preprocessed_block_false])
)


"""
Test `remove_punctuations_s`
"""

text = "hello."
text_preprocessed = "hello "

assert preprocessing.remove_punctuations_s(pd.Series([text])).equals(
    pd.Series([text_preprocessed])
)

"""
Test `remove_diacritics_s`
"""

text = "hèllo"
text_preprocessed = "hello"

assert preprocessing.remove_diacritics_s(pd.Series([text])).equals(
    pd.Series([text_preprocessed])
)


"""
Test `remove_spaces_s`
"""

text = "hello   world  hello        world "
text_preprocessed = "hello world hello world"

assert preprocessing.remove_spaces_s(pd.Series([text])).equals(
    pd.Series([text_preprocessed])
)
