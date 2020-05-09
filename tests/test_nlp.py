from texthero import nlp
import pandas as pd

"""
Test named_entity_recognition
"""

s = pd.Series("New York is a big city")
s_true = pd.Series(['New York', 'LOC', 0, 7])
s_true.equals(nlp.named_entities(s_true))


"""
Test noun_chunks
"""

s = pd.Series("Today is such a beautiful day")
s_true = pd.Series([('Today', 'NP', 0, 5), ('such a beautiful day', 'NP', 9, 29)])
s_true.equals(nlp.named_entities(s_true))
