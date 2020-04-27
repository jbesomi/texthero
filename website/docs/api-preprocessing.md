---
id: api-preprocessing 
title: Preprocessing
---

# Preprocessing

Preprocess text-based Pandas DataFrame.


### texthero.preprocessing.clean(s, pipeline=None)
Clean pandas series by appling a preprocessing pipeline.

For information regarding a specific function type help(texthero.preprocessing.func_name).
The default preprocessing pipeline is the following:

> 
> * fillna


> * lowercase


> * remove_digits


> * remove_punctuation


> * remove_diacritics


> * remove_stop_words


> * remove_whitespace


* **Return type**

    `Series`



### texthero.preprocessing.do_stemm(input, stem='snowball')
Stem series using either NLTK ‘porter’ or ‘snowball’ stemmers.

Not in the default pipeline.


* **Parameters**

    
    * **input** (`Series`) – 


    * **stem** – Can be either ‘snowball’ or ‘stemm’



* **Return type**

    `Series`



### texthero.preprocessing.fillna(input)
Replace not assigned values with empty spaces.


* **Return type**

    `Series`



### texthero.preprocessing.get_default_pipeline()
Return a list contaning all the methods used in the default cleaning pipeline.

Return a list with the following function

    
    * fillna


    * lowercase


    * remove_digits


    * remove_punctuation


    * remove_diacritics


    * remove_stop_words


    * remove_whitespace


* **Return type**

    []



### texthero.preprocessing.lowercase(input)
Lowercase all text.


* **Return type**

    `Series`



### texthero.preprocessing.remove_diacritics(input)
Remove all diacritics.


* **Return type**

    `Series`



### texthero.preprocessing.remove_digits(input, only_blocks=True)
Remove all digits from a series and replace it with a single space.


* **Parameters**

    
    * **input** (*pd.Series*) – 


    * **only_blocks** (*bool*) – Remove only blocks of digits. For instance, hel1234lo 1234 becomes hel1234lo.


### Examples

```python
>>> import texthero
>>> import pandas as pd
>>> s = pd.Series(["texthero 1234 He11o"])
>>> texthero.preprocessing.remove_digits(s)
0    texthero He11o
dtype: object
>>> texthero.preprocessing.remove_digits(s, only_blocks=False)
0    texthero   He o
dtype: object
```


* **Return type**

    `Series`



### texthero.preprocessing.remove_punctuation(input)
Remove string.punctuation (!”#$%&’()\*+,-./:;<=>?@[]^_\`{|}~).

Replace it with a single space.


* **Return type**

    `Series`



### texthero.preprocessing.remove_stop_words(input)
Remove all stop words using NLTK stopwords list.

List of stopwords: NLTK ‘english’ stopwords, 179 items.


* **Return type**

    `Series`



### texthero.preprocessing.remove_whitespace(input)
Remove all white spaces between words.


* **Return type**

    `Series`
