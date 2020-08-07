"""
The texthero.nlp module supports common NLP tasks such as named_entities, noun_chunks, ... on Pandas Series and DataFrame.
"""

import spacy
import pandas as pd


def named_entities(s: pd.Series, package="spacy") -> pd.Series:
    """
    Return named-entities.

    Return a Pandas Series where each row contains a list of tuples
    with information about the named entities in the row's document.

    Tuple: (`entity'name`, `entity'label`, `starting character`, `ending character`)

    Under the hood, `named_entities` makes use of 
    `Spacy name entity recognition <https://spacy.io/usage/linguistic-features#named-entities>`_

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


def noun_chunks(s: pd.Series) -> pd.Series:
    """
    Return noun chunks (noun phrases).

    Return a Pandas Series where each row contains a tuple that has information
    regarding the noun chunk.

    Tuple: (`chunk'text`, `chunk'label`, `starting index`, `ending index`)

    Noun chunks or noun phrases are phrases that have noun at their head or nucleus 
    i.e., they ontain the noun and other words that describe that noun. 
    A detailed explanation on noun chunks: https://en.wikipedia.org/wiki/Noun_phrase
    Internally `noun_chunks` makes use of Spacy's dependency parsing:
    https://spacy.io/usage/linguistic-features#dependency-parse

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


def pos_tag(s: pd.Series) -> pd.Series:
    """
    Return a Pandas Series with part-of-speech tagging

    Return new Pandas Series where each rows contains a list of tuples containing information about part-of-speech tagging

    Tuple (`token name`,`Coarse-grained POS`,`Fine-grained POS`, `starting character`, `ending character`)
    
    A difference between the coarse-grained POS and the Fine-grained POS is that the last one is more specific about marking,
    for example if the coarse-grained POS has a NOUN value, then the refined POS will give more details about the type of 
    the noun, whether it is singular, plural and/or proper.
    You can use the spacy `explain` function to find out which fine-grained POS it is.
    
    You can see more details about Fine-grained POS at: <https://spacy.io/api/annotation#pos-en>

    This makes use of the SpaCy `processing pipeline. <https://spacy.io/usage/processing-pipelines#pipelines>`.

    List of POS/Tag:
     - `ADJ`: Adjective. Examples: big, old, green.
     - `ADP`: Adposition. Examples: in, to, during.
     - `ADV`: Adverb. Examples: very, tomorrow, down.
     - `AUX` : Auxiliary. Examples: is, has (done), will (do).
     - `CONJ`: Conjunction. Examples: and, or, but.
     - `CCONJ`: Coordinating Conjunction. Examples: and, or, but.
     - `DET`: Determiner. Examples: a, an, the.
     - `INTJ`: Interjection. Examples: psst, ouch, bravo.
     - `NOUN`:  Noun. Examples: girl, cat, tree.
     - `NUM`: Numeral. Examples: 1, 2007, one.
     - `PART`: Particle. Examples: 's, not.
     - `PRON`: Pronoun. Examples: I, you, he, she.
     - `PROPN`: Proper Noun. Examples: Mary, John, London.
     - `PUNCT`: Punctuation. Examples: ., (, ), ?
     - `SCONJ`: Subordinating Conjunction. Examples: if, while, that.
     - `SYM`: Symbol. Examples: $, %, §, ©.
     - `VERB`: Verb. Examples: run, runs, running.
     - `X`: Other.
     - `SPACE`: Space.

    Internally pos_tag makes use of Spacy's dependency tagging: <https://spacy.io/api/annotation#pos-tagging>`

    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series("Today is such a beautiful day")
    >>> print(hero.pos_tag(s)[0])
    [('Today', 'NOUN', 'NN', 0, 5), ('is', 'AUX', 'VBZ', 6, 8), ('such', 'DET', 'PDT', 9, 13), ('a', 'DET', 'DT', 14, 15), ('beautiful', 'ADJ', 'JJ', 16, 25), ('day', 'NOUN', 'NN', 26, 29)]
    """

    pos_tags = []

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # nlp.pipe is now "tagger"

    for doc in nlp.pipe(s.astype("unicode").values, batch_size=32):
        pos_tags.append(
            [
                (token.text, token.pos_, token.tag_, token.idx, token.idx + len(token))
                for token in doc
            ]
        )

    return pd.Series(pos_tags, index=s.index)
