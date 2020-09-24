---
id: getting-started-nlp
title: Getting Started - NLP
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
            name                                       history_text                                        powers_text
0        3-D Man  Delroy Garrett, Jr. grew up to become a track ...                                                NaN
1  514A (Gotham)  He was one of the many prisoners of Indian Hil...                                                NaN
2         A-Bomb   Richard "Rick" Jones was orphaned at a young ...    On rare occasions, and through unusual circu...

```

As you can see, we are working with a dataset that's about superheroes! It features each hero's name, a text about their history, and a text describing their superpowers. Of course, all of these can be missing (i.e. "NaN"). We will now try to generate some insights with each of the Texthero NLP functions.

## Count Sentences

### Who is the most well-known superhero?

First of all, we want to know which superhero is the most important. We use the naive approach of counting the number of sentences in their history. The idea is that more well-known superheroes have a richer backstory and writers put more effort into their history.

To count the number of sentences, we use Texthero's `count_sentences` function.

```python
>>> # First, fill the missing values with empty strings.
>>> df["history_text"] = df["history_text"].pipe(hero.fillna)
>>> # Now calculate the number of sentences for each text.
>>> df["history_length"] = df["history_text"].pipe(hero.count_sentences)
>>> df.head(3)
            name                                       history_text                                        powers_text  history_length
0        3-D Man  Delroy Garrett, Jr. grew up to become a track ...                                                NaN               5
1  514A (Gotham)  He was one of the many prisoners of Indian Hil...                                                NaN              38
2         A-Bomb   Richard "Rick" Jones was orphaned at a young ...    On rare occasions, and through unusual circu...              51
```

We now have the number of sentences of the histories. Let's see whose is the longest. We can use Pandas built-in sorting function for that.

```python
>>> # Use pandas built-in sorting to sort by history_length
>>> df.sort_values("history_length", ascending=False, inplace=True)
>>> df.head(5)
                    name                                       history_text                                        powers_text  history_length
1195  Sonic The Hedgehog  Past Not much is known about Sonic's early lif...  Superhuman Speed Sonic's greatest strength is ...            1006
1415           Wolverine    Wolverine's life began in Alberta, Canada, s...     Wolverine is a mutant who has been given an...             652
1421        Wonder Woman  Origin  Wonder Woman did not keep her identity...    Directly after being sculpted from clay, sev...             579
1072           Red Robin   Red Robin is a vigilante and member of the Ba...       Tim Drake has trained under Batman for ye...             578
1098           Robin III   Tim Drake is a vigilante and member of the Ba...     Tim Drake has trained under Batman for year...             514
```

Looks like Sonic has quite the history! We can definitely see that the more well-known heroes are now at the top.


## Noun Chunks

### Find alternative names for the superheroes

We'll now try to find alternative names for our superheroes. For that, we'll use Texthero's `noun_chunks`. The function extracts noun chunks (i.e. chunks of words including a noun and surrounding words that describe that noun) from each text. For example, the sentence "this is a great lake" has the noun chunk "a great lake".

```python
>>> # First, fill the missing values and remove unnecessary whitespace.
>>> df["powers_text"] = df["powers_text"].pipe(hero.fillna).pipe(hero.remove_whitespace)
>>> # Now calculate the noun chunks.
>>> df["noun chunks"] = df["powers_text"].pipe(hero.noun_chunks)
>>> df.head(3)[["name", "powers_text", "noun chunks"]]
                    name                                        powers_text                                        noun chunks
1195  Sonic The Hedgehog  Superhuman Speed Sonic's greatest strength is ...  [(Superhuman Speed Sonic's greatest strength, ...
1415           Wolverine  Wolverine is a mutant who has been given an un...  [(Wolverine, NP, 0, 9), (a mutant, NP, 13, 21)...
1421        Wonder Woman  Directly after being sculpted from clay, sever...  [(clay, NP, 35, 39), (several Olympian gods, N...
```

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
                          name      alternative name                                        powers_text                                        noun chunks
486                  Flash III    the fastest beings  While all speedsters are powered by the force,...  [(all speedsters, NP, 6, 20), (the force, NP, ...
407   Doctor Strange (Classic)  the Sorcerer Supreme  Dr. Strange is the Sorcerer Supreme of Earth's...  [(Dr. Strange, NP, 0, 11), (the Sorcerer Supre...
421                    Dracula       the true master  Passive Attributes Summoning his Demon Castle:...  [(Passive Attributes, NP, 0, 18), (his Demon C...
1270                    Thanos   a superhuman mutant  By far the strongest and most powerful Titania...  [(Thanos, NP, 57, 63), (a superhuman mutant, N...
750                King Thanos   a superhuman mutant  I could not find no powers with King Thanos ex...  [(I, NP, 0, 1), (no powers, NP, 17, 26), (King...
570                     Harumi         the Quiet One  Princess Harumi (also known as the Quiet One, ...  [(Princess Harumi, NP, 0, 15), (the Quiet One,...
```

Looks like we got some good results for those superheroes!

## POS Tagging

### What are the heroes' powers?

In the `powers_text` column, we only get a text describing our heroes' powers. It would be nice to have an easy-to-handle list of their superpowers. For that, we can use _Part-of-Speech Tagging_. This means that we assign each word to a part of speech (e.g. adjective, noun, ...). The adjectives we find could then be potential superpowers.

```python
>>> # Calculate the POS tags.
>>> df["pos tag"] = df["powers_text"].pipe(hero.pos_tag)
>>> df.head(3)[["name", "powers_text", "pos tag"]]
                    name                                        powers_text                                            pos tag
1195  Sonic The Hedgehog  Superhuman Speed Sonic's greatest strength is ...  [(Superhuman, PROPN, NNP, 0, 10), (Speed, PROP...
1415           Wolverine  Wolverine is a mutant who has been given an un...  [(Wolverine, PROPN, NNP, 0, 9), (is, AUX, VBZ,...
1421        Wonder Woman  Directly after being sculpted from clay, sever...  [(Directly, ADV, RB, 0, 8), (after, ADP, IN, 9...
```

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
                          name                                            pos tag                                             powers
486                  Flash III  [(While, SCONJ, IN, 0, 5), (all, DET, DT, 6, 9...  [fastest, fastest, fast, enough, several, own,...
407   Doctor Strange (Classic)  [(Dr., PROPN, NNP, 0, 3), (Strange, PROPN, NNP...  [unparalleled, mystic, otherworldly, primary, ...
421                    Dracula  [(Passive, PROPN, NNP, 0, 7), (Attributes, PRO...  [true, immortal, premature, uncommon, prematur...
1270                    Thanos  [(By, ADP, IN, 0, 2), (far, ADV, RB, 3, 6), (t...  [strongest, powerful, superhuman, massive, hea...
750                King Thanos  [(I, PRON, PRP, 0, 1), (could, VERB, MD, 2, 7)...  [younger, strongest, powerful, superhuman, mas...
570                     Harumi  [(Princess, PROPN, NNP, 0, 8), (Harumi, PROPN,...  [adoptive, close, true, soulless, former, succ...
```


## Named Entities

### Where do our superheroes live?

Having found out so much about our superheroes, we're now interested in where they live. To find that out, we use `hero.named_entities` to find each history text's _Named Entities_. Those are exactly what the name suggests - the entities, e.g. "Yesterday" (a date), "New York" (a location), "Dracula" (a person). We're interested in locations. Those get the tag "GPE" (geographical entity). Thus, we'll first use `named_entities` to get a list of named entities for each row, and then apply a function to extract the most-mentioned geographical entity from the named entities.

```python
>>> # Calculate the Named Entities.
>>> df["named entities"] = df["history_text"].pipe(hero.named_entities)
>>> df.head(3)[["name", "history_text", "named entities"]]
                    name                                       history_text                                     named entities
1195  Sonic The Hedgehog  Past Not much is known about Sonic's early lif...  [(Sonic, ORG, 29, 34), (Christmas Island, LOC,...
1415           Wolverine    Wolverine's life began in Alberta, Canada, s...  [(Wolverine, ORG, 2, 11), (Alberta, GPE, 28, 3...
1421        Wonder Woman  Origin  Wonder Woman did not keep her identity...  [(first, ORDINAL, 76, 81), (Diana, PERSON, 197...
```

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
                    name     location                                     named entities
1195  Sonic The Hedgehog     Robotnik  [(Sonic, ORG, 29, 34), (Christmas Island, LOC,...
1415           Wolverine      Phoenix  [(Wolverine, ORG, 2, 11), (Alberta, GPE, 28, 3...
1421        Wonder Woman        Diana  [(first, ORDINAL, 76, 81), (Diana, PERSON, 197...
1072           Red Robin  Gotham City  [(Red Robin, PERSON, 1, 10), (the Batman Famil...
1098           Robin III      Batcave  [(Tim Drake, PERSON, 1, 10), (the Batman Famil...
```

We get some good and some not-so-good results. There's certainly a lot more fun that can be had with this dataset!

## Recap

In this tutorial, we took a look at all of Texthero's core NLP functions (which are always being expanded and improved). Hopefully you've learned that:
- working with Texthero is really easy,
- Texthero supports the whole NLP workflow, from preprocessing to finding the superpowers of your favorite superheroes,
- the combination of Pandas built-in functions and Texthero's specialised toolset is really powerful.

