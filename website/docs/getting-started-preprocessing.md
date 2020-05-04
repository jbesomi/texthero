---
id: getting-started-preprocessing
---

## Getting started with pre-processing



### Stemming

`do_stem` returns better results when used after `remove_punctuation`.

Example:

```python

>>> text = "I love climbing and running."
>>> hero .stem(pd.Series(text), stem="snowball")
   0    i love climb and running.
   dtype: object
```

Whereas 

```python

>>> text = "I love climbing and running"
>>> hero .stem(pd.Series(text), stem="snowball")
   0    i love climb and run
   dtype: object
```
