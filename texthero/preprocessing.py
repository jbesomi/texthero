import re
import string
import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def remove_lowercase(input):
    """Lowercase input"""
    return input.lower()

def remove_numbers(input):
    """Remove numbers from input"""
    return re.sub(r"\d+", "", input)

def remove_punctuations(input):
    """Remove punctuations from input"""
    return input.translate(str.maketrans("", "", string.punctuation))

def remove_diacritics(input):
    """Remove diacritics (as accent marks) from input"""
    return unidecode.unidecode(input)

def remove_white_space(input):
    """Remove all types of spaces from input"""
    input = input.replace(u"\xa0", u" ")  # remove space
    # remove white spaces, new lines and tabs
    return " ".join(input.split())

def remove_stop_words(input):
    """Remove stopwords from input"""
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(input)
    return [i for i in words if not (i in stop_words)]

def stemming(input):
    """Stem words"""
    stemmer = PorterStemmer()

    if type(input) == str:
        words = word_tokenize(input)
    else:
        words = input

    return " ".join([stemmer.stem(word) for word in words])

def get_default_pipeline():
    """
    Default pipeline:
        - remove_lowercase
        - remove_numbers
        - remove_punctuations
        - remove_diacritics
        - remove_white_space
        - remove_stop_words
        - stemming
    """
    return [remove_lowercase,
            remove_numbers,
            remove_punctuations,
            remove_diacritics,
            remove_white_space,
            remove_stop_words,
            stemming]

def apply_fun_to_obj(fun, obj, text_columns):
    for col in text_columns:
        obj[col] = fun(obj[col])
    return obj

def do_preprocess(df, text_columns=['text'], pipeline=None):

    if not pipeline:
        pipeline = get_default_pipeline()

    def clean(text):
        if text is None:
            return text
        for f in pipeline:
            text = f(text)
        return text

    if isinstance(text_columns, str):
        df[text_columns + "_clean"] = df.apply(lambda row: clean(row[text_columns]), axis=1)
    else:
        for col in text_columns:
            df[col + "_clean"] = df.apply(lambda row: clean(row[col]), axis=1)

    return df
