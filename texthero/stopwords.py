import nltk
import spacy

try:
    # If not present, download NLTK stopwords.
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords as nltk_en_stopwords

try:
    # If not present, download 'en_core_web_sm'
    spacy_model = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli.download import download as spacy_download

    spacy_download("en_core_web_sm")

from spacy.lang.en import stop_words as spacy_en_stopwords

DEFAULT = set(nltk_en_stopwords.words("english"))
NLTK_EN = DEFAULT
SPACY_EN = spacy_en_stopwords.STOP_WORDS
