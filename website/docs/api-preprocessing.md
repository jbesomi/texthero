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



### texthero.preprocessing.do_stem(input, stem='snowball', language='english')
Stem series using either ‘porter’ or ‘snowball’ NLTK stemmers.

Not in the default pipeline.


* **Parameters**

    
    * **input** (`Series`) – 


    * **stem** – Can be either ‘snowball’ or ‘porter’. (“snowball” is default)


    * **language** – Supportted languages:

        danish dutch english finnish french german hungarian italian
        norwegian porter portuguese romanian russian spanish swedish




* **Return type**

    `Series`



### texthero.preprocessing.drop_no_content(s)
Drop all rows where has_content is empty.

### Example

```python
>>> s = pd.Series(["c", np.nan, "   
", " "])
>>> drop_no_content(s)
0    c
dtype: object
```


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



### texthero.preprocessing.has_content(s)
For each row, check that there is content.

### Example

```python
>>> s = pd.Series(["c", np.nan, "   
", " "])
>>> has_content(s)
0     True
1    False
2    False
3    False
dtype: bool
```


### texthero.preprocessing.lowercase(input)
Lowercase all text.


* **Return type**

    `Series`



### texthero.preprocessing.remove_angle_brackets(s)
Remove content within angle brackets <> and the angle brackets.

### Example

```python
>>> s = pd.Series("Texthero <is not a superhero!>")
>>> remove_angle_brackets(s)
0    Texthero
dtype: object
```


### texthero.preprocessing.remove_brackets(s)
Remove content within brackets and the brackets.

Remove content from any kind of brackets, (), [], {}, <>.

### Example

```python
>>> s = pd.Series("Texthero (round) [square] [curly] [angle]")
>>> remove_brackets(s)
0    Texthero
dtype: object
```


### texthero.preprocessing.remove_curly_brackets(s)
Remove content within curly brackets {} and the curly brackets.

### Example

```python
>>> s = pd.Series("Texthero {is not a superhero!}")
>>> remove_curly_brackets(s)
0    Texthero
dtype: object
```


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
>>> s = pd.Series("7ex7hero is fun 1111")
>>> remove_digits(s)
0    7ex7hero is fun
dtype: object
>>> remove_digits(s, only_blocks=False)
0    exhero is fun
dtype: object
```


* **Return type**

    `Series`



### texthero.preprocessing.remove_punctuation(input)
Remove string.punctuation (!”#$%&’()\*+,-./:;<=>?@[]^_\`{|}~).

Replace it with a single space.


* **Return type**

    `Series`



### texthero.preprocessing.remove_round_brackets(s)
Remove content within parentheses () and parentheses.

### Example

```python
>>> s = pd.Series("Texthero (is not a superhero!)")
>>> remove_round_brackets(s)
0    Texthero
dtype: object
```


### texthero.preprocessing.remove_square_brackets(s)
Remove content within square brackets [] and the square brackets.

### Example

```python
>>> s = pd.Series("Texthero [is not a superhero!]")
>>> remove_square_brackets(s)
0    Texthero
dtype: object
```


### texthero.preprocessing.remove_stop_words(input)
Remove all stop words using NLTK stopwords list.

List of stopwords: NLTK ‘english’ stopwords, 179 items.


* **Return type**

    `Series`



### texthero.preprocessing.remove_whitespace(input)
Remove all white spaces between words.


* **Return type**

    `Series`
