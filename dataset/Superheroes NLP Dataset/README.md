# Superheroes NLP Dataset

A playground dataset to learn and practice NLP, text mining and data analysis while having fun.

The same dataset can be found on Kaggle: [Superheroes NLP Dataset](https://www.kaggle.com/jonathanbesomi/superheroes-nlp-dataset).

All data have been scraped with python from [Superhero Database](https://www.superherodb.com/), credits belongs to them.

## Dataset summary

Size: 8 MB.

Num. columns: 81.

Num. superheroes: 1447.

Main columns:
   - name
   - real_name
   - full_name
   - overall_score - how powerful is the superhero according to superherodb.
   - *history_text* - Superhero's history.
   - *powers_text* - Description of superhero's powers
   - intelligence_score
   - strength_score
   - speed_score
   - durability_score
   - power_score	
   - combat_score
   - alter_egos - List of alternative personality
   - aliases 
   - creator - _DC Comics_ or _Marvel Comics_ for instance.
   - alignment	- Is the character good or bad?
   - occupation
   - type_race	
   - height	
   - weight	
   - eye_color	
   - hair_color	
   - skin_color


## Getting started

You can download the complete dataset directly from Github here: [Superheroes NLP Dataset](https://github.com/jbesomi/texthero/tree/master/dataset/Superheroes%20NLP%20Dataset/data).

If you feel lazy, you can also import it directly from pandas:

```python
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/jbesomi/texthero/master/dataset/superheroes_nlp_dataset.csv")

df.head()
```

```bash
            name               real_name               full_name overall_score  ... has_durability has_stamina  has_agility  has_super_strength
0        3-D Man     Delroy Garrett, Jr.     Delroy Garrett, Jr.             6  ...            0.0         0.0          0.0                 1.0
1  514A (Gotham)             Bruce Wayne                     NaN            10  ...            1.0         0.0          0.0                 1.0
2         A-Bomb  Richard Milhouse Jones  Richard Milhouse Jones            20  ...            1.0         1.0          1.0                 1.0
3             Aa                      Aa                     NaN            12  ...            0.0         0.0          0.0                 0.0
4     Aaron Cash              Aaron Cash              Aaron Cash             5  ...            0.0         0.0          0.0                 0.0

[5 rows x 81 columns]
```
