# Superheroes NLP Dataset

A playground dataset to learn and practice NLP, text mining and data analysis while having fun.

The same dataset can be found on Kaggle: [Superheroes NLP Dataset]().

All data have been scraped with python from [Superhero Database](https://www.superherodb.com/), credits belongs to them.

## Dataset summary

Size: 10 MB.
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

You can download the complete dataset directly from Github here: [Superheroes NLP Dataset](./dataset/Superheroes NLP Dataset/data/superheroes_nlp_dataset.csv).

If you feel lazy, you can also import it directly from pandas:

```python
import pandas as pd

df = df.read_csv("")

df.head()
```
