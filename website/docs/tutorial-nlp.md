---
id: tutorial-nlp
title: Tutorial NLP
---

# Texthero's NLP Module

Texthero's NLP module features many common _Natural Language Processing_ functions applied to Pandas Series. You can see all functions with a detailed description and examples [here](https://texthero.org/docs/api-nlp). In this tutorial, we'll have a quick look at some of the functions and apply them to a real dataset.

## Load Data and Preprocess
Let's begin by loading an interesting dataset and having a first look.
```python
>>> import texthero as hero
>>> import pandas as pd
>>> df = pd.read_csv("https://raw.githubusercontent.com/jbesomi/texthero/master/dataset/superheroes_nlp_dataset.csv")
>>> # We only keep a few interesting columns.
>>> df = df[["name", "history_text", "powers_text"]]
>>> df.head(3)
```
![](./assets/tutorial-nlp-S1.png)

As you can see, we are working with a dataset that's about superheroes! It features each hero's name, a texth about their history, and a text describing their superpowers. Of course, all of these can be missing (i.e. "NaN"). We will now try to generate some insights with each of the Texthero NLP functions.

## Count Sentences - Who is the most well-known superhero?
First of all, we want to know which superhero is the most important. We use the naive approach of counting the number of sentences in their history. The idea is that more well-known superheroes have a richer backstory and writers put more effort into their history.

Of course, we use Texthero's `count_sentences` function.

```python
>>> # First, fill the missing values with empty strings.
>>> df["history_text"] = df["history_text"].pipe(hero.fillna)
>>> # Now calculate the number of sentences for each text.
>>> df["history_length"] = df["history_text"].pipe(hero.count_sentences)
>>> df.head(3)
```
![](./assets/tutorial-nlp-S2.png)

We now have the number of sentences of the histories. Let's see whose is the longest. We can use Pandas built-in sorting function for that.

```python
>>> # Use pandas built-in sorting to sort by history_length
>>> df.sort_values("history_length", ascending=False, inplace=True)
>>> df.head(5)
```
![](./assets/tutorial-nlp-S3.png)

Looks like Sonic has quite the history! We can definitely see that the more well-known heroes are now at the top.


## Noun Chunks - Find alternative names for the superheroes

We'll now try to find alternative names for our superheroes. For that, we'll use Texthero's `noun_chunks`. The function extracts noun chunks (i.e. chunks of words including a noun and surrounding words that describe that noun) from each text. For example, the sentence "this is a great lake" has the noun chunk "a great lake".

```python
>>> # First, fill the missing values and remove whitespace.
>>> df["powers_text"] = df["powers_text"].pipe(hero.fillna).pipe(hero.remove_whitespace)
>>> # Now calculate the noun chunks.
>>> df["noun chunks"] = df["powers_text"].pipe(hero.noun_chunks)
>>> df.head(3)[["name", "powers_text", "noun chunks"]]
```
![](./assets/tutorial-nlp-S4.png)

To get alternative names, we now loop through every row in the `noun chunks` field and extract the first noun chunk with length 3 that starts with "a" or "the" - the hope is to extract stuff like "the green mutant" for Hulk. In pandas, this is really easy: We write a function that works on one list of noun chunks (i.e. one cell) and then use `apply` to apply that function to a whole column.

Here is the function:

```python
def alternative_name_from_noun_chunks(list_of_noun_chunks):
    # Loop through the chunks.
    for (chunk, _, _, _) in list_of_noun_chunks:
        if (chunk.startswith("the ") or chunk.startswith("a ")) and len(chunk.split()) == 3:
            return chunk
    # Don't find a potential alternative name -> return NaN.
    return pd.NA
```

Now just apply:

```python
>>> df["alternative name"] = df["noun chunks"].apply(alternative_name_from_noun_chunks)
```


Let's have a look at some selected alternative names (of course this does not work perfectly for all superheros). Results are good for e.g. Flash, Thanos, Doctor Strange, Dracula, and Harumi, so we look at those.

```python
>>> # First fill missing values with empty strings.
>>> df["name"] = df["name"].pipe(hero.fillna)
>>> # Now, use pandas `.str.contains` method to get the indexes of the interesting rows.
>>> interesting_rows = df["name"].str.contains('Flash III|Doctor Strange|Dracula|Thanos|Harumi')
>>> # Finally, look at the interesting rows.
>>> df[interesting_rows][["name", "alternative name", "powers_text", "noun chunks"]]
```
![](./assets/tutorial-nlp-S5.png)

Looks like we got some good results for those superheroes!

## POS Tagging - What are the heroes' powers?

In the `powers_text` column, we only get a text describing our heroes' powers. It would be nice to have an easy-to-handle list of their superpowers. For that, we can use _Part-of-Speech Tagging_. This means that we assign each word to a part of speech (e.g. adjective, noun, ...). The adjectives we find could then be potential superpowers.

```python
>>> # Calculate the POS tags.
>>> df["pos tag"] = df["powers_text"].pipe(hero.pos_tag)
>>> df.head(3)[["name", "powers_text", "pos tag"]]
```
![](./assets/tutorial-nlp-S6.png)

Just like with the noun chunks, we now extract the adjectives by writing a function that extracts them from a list of POS-tags and applying that function to the whole column.

```python
def adjectives_from_pos_tags(list_of_pos_tags):
    # Return a list of all words whose part-of-speech is "ADJ", so all adjectives.
    return [word for (word, kind, _, _, _) in list_of_pos_tags if kind == "ADJ"]
```

Again, just apply:

```python
>>> df["powers"] = df["pos tag"].apply(adjectives_from_pos_tags)
>>> # Look at the interesting rows we defined above again.
>>> df[interesting_rows][["name", "pos tag", "powers"]]
```
![](./assets/tutorial-nlp-S7.png)


## Named Entities - Where do our superheroes live?

Having found out so much about our superheroes, we're now interested in where they live. To find that out, we use `hero.named_entities` to find each history text's _Named Entities_. Those are exactly what the name suggests - the entities, e.g. "Yesterday" (a date), "New York" (a location), "Dracula" (a person). We're interested in locations. Those get the tag "GPE" (geographical entity). Thus, we'll first use `named_entities` to get a list of named entities for each row, and then apply a function to extract the most-mentioned geographical entity from the named entities.

```python
>>> # Calculate the Named Entities.
>>> df["named entities"] = df["history_text"].pipe(hero.named_entities)
>>> df.head(3)[["name", "history_text", "named entities"]]
```
![](./assets/tutorial-nlp-S8.png)

Here's the function to extract the most common geographical entity:

```python
def location_from_named_entities(list_of_named_entities):
    # Collect all geographical entities.
    mentioned_locations = [
        entity for (entity, label, _, _) in list_of_named_entities if label == "GPE"
    ]
    # If any were found, return the most common one.
    if mentioned_locations:
        most_frequently_mentioned_location = max(
            mentioned_locations,
            key=mentioned_locations.count
        )
        return most_frequently_mentioned_location
    else:
        return ""
```

Let's apply the function and take a look at a few results.

```python
>>> df["location"] = df["named entities"].apply(location_from_named_entities)
>>> df[["name", "location", "named entities"]].head(5)
```
![](./assets/tutorial-nlp-S9.png)

We get some good and some not-so-good results. There's certainly a lot more fun that can be had with this dataset!

## Recap

In this tutorial, we took a look at all of Texthero's core NLP functions (which are always being expanded and improved). Hopefully you've learned that

- working with Texthero is really easy,
- Texthero supports the whole NLP workflow, from preprocessing to finding the superpowers of your favorite superheroes,
- the combination of Pandas built-in functions and Texthero's specialised toolset is really powerful.

