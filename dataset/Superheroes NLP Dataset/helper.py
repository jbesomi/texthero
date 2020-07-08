"""
Helper functions to scrape all superrheroes data from 'https://www.superherodb.com/'
"""

from bs4 import BeautifulSoup
import urllib3
import pandas as pd
from collections import defaultdict
import requests
import re


def get_data(url):
    """
    Return BeautifulSoup html object.
    """
    r = requests.get(url)
    data = BeautifulSoup(r.text, "lxml")
    return data


def get_superheroes_links(data):
    herolinks = []

    home_url = "https://www.superherodb.com"

    for all_li in data.find_all(class_="list"):
        for link in all_li.find_all("li"):
            for hero in link.find_all("a"):
                herolinks.append(home_url + hero["href"])
    return herolinks


def get_id_from_about(filename):
    """
    Extract id from local filename.
    """
    return filename.replace("_about.html", "").split("/")[-1]


def get_soup(filename):
    with open(filename, "rb") as f:
        file = f.read()
        return BeautifulSoup(file, "lxml")


"""
Get data
"""


def get_data(url):
    r = requests.get(url)
    data = BeautifulSoup(r.text, "lxml")
    return data


"""
About
"""


def get_image(data_about):

    img = data_about.find(class_="portrait").find("img")
    if img:
        return dict(img=img["src"])
    else:
        return dict(img=None)


def get_name_real_name(data_about):
    name = data_about.find("h1").text
    real_name = data_about.find("h2").text
    return dict(name=name, real_name=real_name)


def get_overall_score(data_about):
    return dict(overall_score=data_about.find(href="#class-info").text)


def get_power_stats(data_about):

    scripts = data_about.findAll("script")
    # Find script containng the 'stats_shdb'
    script = next(
        (s.text for s in scripts if s.text.strip().startswith("var stats_shdb = ["))
    )
    # Extract the list of powers
    values = re.findall(r"(\d+)", script.split(";")[0])
    values = [int(v) for v in values]

    labels = data_about.find(class_="stat-holder").findAll("label")
    labels = [l.text for l in labels]

    return dict(zip(labels, values))


def get_super_powers(data_about):
    superpowers = data_about.find("h3", text="Super Powers").findParent().findAll("a")
    superpowers = [s.text for s in superpowers]
    return dict(superpowers=superpowers)


def get_all_links(td):
    links = td.findAll("a")
    links = [a.text for a in links]
    return links


def get_origin(data_about):

    data = data_about.find("h3", text="Origin").findNext()

    origin = {}

    for row in data.find_all("tr"):
        key = row.find_all("td")[0].text
        value = row.find_all("td")[1]

        if "alter egos" in key.lower():
            origin[key] = get_all_links(value)
        else:
            origin[key] = value.text
    return origin


def get_connections(data_about):
    data = data_about.find("h3", text="Connections").findNext()

    connections = {}

    for row in data.find_all("tr"):
        key = row.find_all("td")[0].text
        value = row.find_all("td")[1]

        if "Teams" in key:
            connections[key] = get_all_links(value)
        else:
            connections[key] = value.text

    return connections


def get_appearance(data_about):
    table = data_about.find("h3", text="Appearance").findParent()
    labels = table.findAll(class_="table-label")
    return dict([(l.text, l.findNext().text) for l in labels])


"""
History
"""


def get_history(data_history):
    content = data_history.find(class_="text-columns-2")
    title = content.find("h3").text
    subtitles = [s.text for s in content.findAll("h4")]
    content = " ".join([p.text for p in content.findAll("p")]).replace("\s+", " ")
    return {"hist_title": title, "hist_subtitles": subtitles, "hist_content": content}


"""
Powers
"""


def get_powers(data_powers):
    content = data_powers.find_all(class_="col-8")[1]
    title = content.find("h3").text
    subtitles = [s.text for s in content.findAll("h4")]
    content = " ".join([p.text for p in content.findAll("p")]).replace("\s+", " ")
    return {
        "powers_title": title,
        "powers_subtitles": subtitles,
        "powers_content": content,
    }


"""
Merge all
"""


def merge_data(data_about, data_history, data_powers):

    data = {}

    # Get from about page
    data.update(get_image(data_about))
    data.update(get_name_real_name(data_about))
    data.update(get_overall_score(data_about))
    data.update(get_power_stats(data_about))
    data.update(get_super_powers(data_about))
    data.update(get_origin(data_about))
    data.update(get_connections(data_about))
    data.update(get_appearance(data_about))

    # Get history data
    data.update(get_history(data_history))

    # Get powers data
    data.update(get_powers(data_powers))

    return data
