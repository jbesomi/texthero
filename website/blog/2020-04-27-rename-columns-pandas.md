---
title: Groupby and rename columns in pandas
author: Jonathan Besomi
unlisted: True
---



## Groupby and rename columns in pandas

```
df.groupby(['artist']).mean().stack().rename_axis(['one', 'bar']).reset_index(name='ooo')
```

```
df_empath = (
    df_empath.groupby(['artist'])
             .max()
             .stack()
             .rename_axis(['artist', 'sentiment'])
             .reset_index(name='r')
)
```
