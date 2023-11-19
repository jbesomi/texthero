---
title: Read Text Data From PDF and use it in Texthero
author: Selim Alawwa
unlisted: True
---

## Introduction
PDF files are widely used due to its broad compatibility however handling or editing its data is a rather uneasy task. Since many reports and documents that contain valuable information are in PDF and analysing this data can be provide high value to users, we thought we can show how we can read PDF data and use them in texthero.

## Working with PDF
Python pandas does not support reading PDF file by default so we will explore the various options and python libraries that enable us to read PDF files and load them into python data frames

## Example using Tabula
tabula-py is a simple Python wrapper of tabula-java, which can read tables from PDF and convert them into pandas’s DataFrame. The library requires having Java 8+ and Python 3.5+.

```
import texthero as hero
try:
    import tabula
except:
    !pip install tabula-py
    import tabula
```

read_pdf funtion returns an array of dataframes obtained from the pdf file.
```
dataframes = tabula.read_pdf("path/to/pdf/file", pages='all')
df = dataframes[0]
df.head()
```
Apply texthero clean method then visualize the a simple word cloud 
```
df['text_clean'] = df['text'].pipe(hero.clean)
df['text_clean'].pipe(hero.wordcloud, max_words=50)
```