"""
Create the dataset from the raw data
"""

import helper as h
import glob
import pandas as pd
from tqdm import tqdm
import ast

DOWNLOAD_DIR = "./data/raw/"

all_files = glob.glob(DOWNLOAD_DIR + "*.html")

# Get unique names:
all_about = glob.glob(DOWNLOAD_DIR + "*_about.html")

ids = [h.get_id_from_about(a) for a in all_about]

dataset = []

for id in tqdm(ids):

    filename_about = DOWNLOAD_DIR + id + "_about.html"
    filename_history = DOWNLOAD_DIR + id + "_history.html"
    filename_powers = DOWNLOAD_DIR + id + "_powers.html"

    try:
        h.get_soup(filename_about)
    except:
        print(filename_about)

    data_about = h.get_soup(filename_about)
    data_history = h.get_soup(filename_history)
    data_powers = h.get_soup(filename_powers)

    row = h.merge_data(data_about, data_history, data_powers)

    dataset.append(row)

df = pd.DataFrame(dataset)

df.columns = df.columns.str.lower()
# Clean dataset


def clean_teams(df):
    df["teams"] = df["teams"].astype(str)
    df["teams"] = df["teams"].str.replace("\nNo teams added.", "no_team")

    df["teams"] = df["teams"].str.replace("\n", "").str.strip()
    return df


df = clean_teams(df)

# lowercase all columns
df.columns = df.columns.str.lower().str.replace(" ", "_")


# Rename columns
df = df.rename(columns={"type_/_race": "type_race"})

power_score = dict(
    intelligence="intelligence_score",
    strength="strength_score",
    speed="speed_score",
    durability="durability_score",
    power="power_score",
    combat="combat_score",
)

df = df.rename(columns=power_score)

df = df.rename(columns=dict(hist_content="history_text", powers_content="powers_text"))

# Reorder columns
df = df[
    [
        "name",
        "real_name",
        "full_name",
        "overall_score",
        "history_text",
        "powers_text",
        "intelligence_score",
        "strength_score",
        "speed_score",
        "durability_score",
        "power_score",
        "combat_score",
        "superpowers",
        "alter_egos",
        "aliases",
        "place_of_birth",
        "first_appearance",
        "creator",
        "alignment",
        "occupation",
        "base",
        "teams",
        "relatives",
        "gender",
        "type_race",
        "height",
        "weight",
        "eye_color",
        "hair_color",
        "skin_color",
        "img",
    ]
]

# Extract 'superpowers' data

df_superpowers = (
    df["superpowers"].apply(pd.Series).stack().pipe(pd.get_dummies).sum(level=0)
)

# Keep only most 50 common superpowers
common_superpowers = df_superpowers.sum(axis=0).sort_values().tail(50).index
df_superpowers = df_superpowers[common_superpowers]
df_superpowers.columns = df_superpowers.columns.str.lower().str.replace(" ", "_")
df_superpowers = df_superpowers.add_prefix("has_")

df = df.join(df_superpowers)


# Split aliases
df["aliases"] = df["aliases"].str.split("\n")

print(df.shape)

# Keep only rows where 'history_text' or 'powers_text' is not null.
df = df[
    ~(df["history_text"].str.strip() == "") | ~(df["powers_text"].str.strip() == "")
]

print(df.shape)


df.to_csv("./data/superheroes_nlp_dataset.csv", index=False)
