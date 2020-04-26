---
id: api-preprocessing 
title: Preprocessing
---

# Preprocessing

Utility functions to clean text-columns of a dataframe.


### texthero.preprocessing.fillna(input)
Replace not assigned values with empty spaces.


* **Return type**

    `Series`



### texthero.preprocessing.get_default_pipeline()
Default pipeline:

    
    * remove_lowercase


    * remove_numbers


    * remove_punctuation


    * remove_diacritics


    * remove_white_space


    * remove_stop_words


    * stemming


### texthero.preprocessing.lowercase(input)
Lowercase all cells.


* **Return type**

    `Series`



### texthero.preprocessing.remove_diacritics(input)
Remove diacritics (as accent marks) from input


* **Return type**

    `Series`



### texthero.preprocessing.remove_digits(input, only_blocks=True)
Remove all digits.


* **Parameters**

    
    * **input** (*pd.Series*) – 


    * **only_blocks** (*bool*) – Remove only blocks of digits. For instance, hel1234lo 1234 becomes hel1234lo.



* **Returns**

    


* **Return type**

    pd.Series


### Examples

```python
>>> import texthero
>>> s = pd.Series(["remove_digits_s remove all the 1234 digits of a pandas series. H1N1"])
>>> texthero.preprocessing.remove_digits_s(s)
u'remove_digits_s remove all the digits of a pandas series. H1N1'
>>> texthero.preprocessing.remove_digits_s(s, only_blocks=False)
u'remove_digits_s remove all the digits of a pandas series. HN'
```


### texthero.preprocessing.remove_punctuation(input)
Remove punctuations from input


* **Return type**

    `Series`



### texthero.preprocessing.remove_whitespaces(input)
Remove any type of space between words.


* **Return type**

    `Series`
