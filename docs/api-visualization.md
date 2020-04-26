---
id: api-visualization 
title: Visualization
---

# Visualization

Text visualization


### texthero.visualization.scatterplot(df, col, color=None, hover_data=None, title='')
Scatterplot of df[column].

The df[column] must be a tuple of 2d-coordinates.

Usage example:

```python
>>> import texthero
>>> df = pd.DataFrame([(0,1), (1,0)], columns='pca')
>>> texthero.visualization.scatterplot(df, 'pca')
```


### texthero.visualization.top_words(s, normalize=True)
Return most common words of a given series sorted from most used.


* **Return type**

    `Series`
