from nltk.corpus import stopwords as nltk_stopwords
from spacy.lang.en import stop_words as spacy_en_stop_words

DEFAULT = set(nltk_stopwords.words("english"))
NLTK_EN = DEFAULT
SPACY_EN = spacy_en_stop_words.STOP_WORDS
