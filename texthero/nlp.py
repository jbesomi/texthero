"""
Common NLP tasks such as named_entities, noun_chunks, etc.
"""

import spacy
import pandas as pd


def named_entities(s: pd.Series, package="spacy") -> pd.Series:
    """
    Return named-entities.

    Return a Pandas Series where each rows contains a list of tuples containing information regarding the given named entities.

    Tuple: (`entity'name`, `entity'label`, `starting character`, `ending character`)

    Under the hood, `named_entities` makes use of `Spacy name entity recognition <https://spacy.io/usage/linguistic-features#named-entities>`_

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

    Parameters
    ----------
        s : Pandas Series

    Returns
    -------
        Pandas Series, where each rows contains a list of tuples containing information regarding the given named entities.

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


def noun_chunks(s: pd.Series) -> pd.Series:
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
        s : Pandas Series

    Returns
    -------
        Pandas Series, where each row contains a tuple that has information regarding the noun chunk.

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

    This makes use of the SpaCy `sentencizer <https://spacy.io/api/sentencizer>`_

    Parameters
    ----------
        s : Pandas Series

    Returns
    -------
        Pandas Series, with the number of sentences per document in every cell.

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
