"""
Common NLP tasks such as named_entities, noun_chunks, etc.
"""

import spacy
import pandas as pd
from spacy_langdetect import LanguageDetector
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from langdetect.language import Language


def named_entities(s, package="spacy"):
    """
    Return named-entities.

    Return a Pandas Series where each rows contains a list of tuples containing information regarding the given named entities.

    Tuple: (`entity'name`, `entity'label`, `starting character`, `ending character`)

    Under the hood, `named_entities` make use of Spacy name entity recognition.

    List of labels:
     - `PERSON`: People, including fictional.
     - `NORP`: Nationalities or religious or political groups.
     - `FAC`: Buildings, airports, highways, bridges, etc.
     - `ORG` : Companies, agencies, institutions, etc.
     - `GPE`: Countries, cities, states.
     - `LOC`: Non-GPE locations, mountain ranges, bodies of water.
     - `PRODUCT`: Objects, vehicles, foods, etc. (Not services.)
     - `EVENT`: Named hurricanes, battles, wars, sports events, etc.
     - `WORK_OF_ART`: Titles of books, songs, etc.
     - `LAW`: Named documents made into laws.
     - `LANGUAGE`: Any named language.
     - `DATE`: Absolute or relative dates or periods.
     - `TIME`: Times smaller than a day.
     - `PERCENT`: Percentage, including ”%“.
     - `MONEY`: Monetary values, including unit.
     - `QUANTITY`: Measurements, as of weight or distance.
     - `ORDINAL`: “first”, “second”, etc.
     - `CARDINAL`: Numerals that do not fall under another type.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Yesterday I was in NY with Bill de Blasio")
    >>> hero.named_entities(s)[0]
    [('Yesterday', 'DATE', 0, 9), ('NY', 'GPE', 19, 21), ('Bill de Blasio', 'PERSON', 27, 41)]
    """
    entities = []

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
    # nlp.pipe is now 'ner'

    for doc in nlp.pipe(s.astype("unicode").values, batch_size=32):
        entities.append(
            [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
        )

    return pd.Series(entities, index=s.index)


def noun_chunks(s):
    """
    Return noun chunks (noun phrases).

    Return a Pandas Series where each row contains a tuple that has information regarding the noun chunk.
    
    Tuple: (`chunk'text`, `chunk'label`, `starting index`, `ending index`)

    Noun chunks or noun phrases are phrases that have noun at their head or nucleus 
    i.e., they ontain the noun and other words that describe that noun. 
    A detailed explanation on noun chunks: https://en.wikipedia.org/wiki/Noun_phrase
    Internally `noun_chunks` makes use of Spacy's dependency parsing:
    https://spacy.io/usage/linguistic-features#dependency-parse

    Parameters
    ----------
    input : Pandas Series
    
    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("The spotted puppy is sleeping.")
    >>> hero.noun_chunks(s)
    0    [(The spotted puppy, NP, 0, 17)]
    dtype: object
    """

    noun_chunks = []

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    # nlp.pipe is now "tagger", "parser"

    for doc in nlp.pipe(s.astype("unicode").values, batch_size=32):
        noun_chunks.append(
            [
                (chunk.text, chunk.label_, chunk.start_char, chunk.end_char)
                for chunk in doc.noun_chunks
            ]
        )

    return pd.Series(noun_chunks, index=s.index)


def count_sentences(s: pd.Series) -> pd.Series:
    """
    Count the number of sentences per cell in a Pandas Series.

    Return a new Pandas Series with the number of sentences per cell.

    This makes use of the SpaCy `sentencizer <https://spacy.io/api/sentencizer>`.

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Yesterday I was in NY with Bill de Blasio. Great story...", "This is the F.B.I.! What? Open up!"])
    >>> hero.count_sentences(s)
    0    2
    1    3
    dtype: int64
    """
    number_of_sentences = []

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    nlp.add_pipe(nlp.create_pipe("sentencizer"))  # Pipe is only "sentencizer"

    for doc in nlp.pipe(s.values, batch_size=32):
        sentences = len(list(doc.sents))
        number_of_sentences.append(sentences)

    return pd.Series(number_of_sentences, index=s.index)


def _Language_to_tuple(lang: Language):
    return (str(lang.lang), "%.5f" % float(lang.prob))


def _detect_language_probability(s):
    """
    gured out appling detect_langs function on sentence
    :param s
    """
    try:
        detected_language = list(map(_Language_to_tuple, detect_langs(s)))
        return detected_language
    except LangDetectException:
        return ("UNKNOWN", 0.0)


def _detect_language(s):
    """
    gured out appling detect_langs function on sentence
    :param s
    """
    try:
        detected_language = str(detect_langs(s)[0].lang)
        return detected_language
    except LangDetectException:
        return "UNKNOWN"


def infer_lang(s, probability=False):
    """
    Return languages and their probabilities.

    Return a Pandas Series where each row contains a ISO nomenclature of the "average" infer language.

    If probability = True then each row contains a list of tuples

    Tuple : (language, probability)

    Note: infer_lang is nondeterministic function

    Parameters
    ----------
    s : Pandas Series
    probability (optional) : boolean

    supports 55 languages out of the box (ISO 639-1 codes)
    ------------------------------------------------------
    af, ar, bg, bn, ca, cs, cy, da, de, el, en, es, et, fa, fi, fr, gu, he,
    hi, hr, hu, id, it, ja, kn, ko, lt, lv, mk, ml, mr, ne, nl, no, pa, pl,
    pt, ro, ru, sk, sl, so, sq, sv, sw, ta, te, th, tl, tr, uk, ur, vi, zh-cn, zh-tw

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("This is an English text!.")
    >>> hero.infer_lang(s)
    0    en
    dtype: object

    """

    if probability:
        return s.apply(_detect_language_probability)
    else:
        return s.apply(_detect_language)
