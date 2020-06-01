"""
Common NLP tasks such as named_entities, noun_chunks, etc.
"""

import spacy
import pandas as pd


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
    Return noun_chunks, group of consecutive words that belong together.
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
