import nltk
import spacy

try:
    # If not present, download NLTK stopwords.
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords as nltk_en_stopwords
from spacy.lang.en import stop_words as spacy_en_stopwords

DEFAULT = set(nltk_en_stopwords.words("english"))
NLTK_EN = DEFAULT
SPACY_EN = spacy_en_stopwords.STOP_WORDS
