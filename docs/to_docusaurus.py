#!/usr/bin/env python3

"""
Takes the output from Sphinx, clean it and send it to Docusaurus.

1. Get four main modules from _build/html/
    - Extract only the 'body' html and store it as a md file under 
            ./website/docs/api-{module_name}.md

2. Get all files under _build/html/api/
    - Extract 'body' html and store it as a md file under
            ./website/docs/api/{filenames}.md

3. Update 'sidebars.json' with the new markdown files
    - Update the 'api' section.
    - Add each module under a sub-directory.
"""


"""
Takes all relevant html files from the html output sphinx folder, parse it with Beautifulsoup, remove unnecessary html data (such as <head>) and
save a markdown file.
"""

from bs4 import BeautifulSoup
import glob
from pathlib import Path
from typing import List
import re
import json

"""
PARAMETERS
"""

MODULES = ["preprocessing", "nlp", "representation", "visualization"]
ROOT_HTML_DIRECTORY = "./_build/html"
ROOT_MD_DIRECTORY = "../website/docs/"
SIDEBARS_FILEPATH = "../website/sidebars.json"
"""
Helper functions
"""


def get_content(soup):
    return soup.find("main").find("div")


def add_docusaurus_metadata(content: str, id: str, title: str, hide_title) -> str:
    """
    Add docusaurus metadata into content.
    """
    return f"---\nid: {id}\ntitle: {title}\nhide_title: {hide_title}\n---\n\n" + content


def fix_href(soup, module: str):
    """
    Fix internal href to be compatible with docusaurus.
    """

    for a in soup.find_all("a", {"class": "reference internal"}, href=True):
        a["href"] = re.sub("^texthero\.", f"/docs/{module}/", a["href"])
        a["href"] = a["href"].lower()
    return soup


def to_md(
    in_html_filepath: str, out_md_filepath: str, id: str, title: str, hide_title: str
) -> None:
    """
    Convert Sphinx-generated html to md.

    Parameters
    ----------
    in_html_filepath : str
        input html file. Example: ./_build/html/preprocessing.html
    out_md_filepath : str
        output html file. Example: ../website/docs/preprocessing.md
    id : str
        Docusaurus document id
    title : str
        Docusaurus title id
    hide_title : str ("true" or "false")
        Whether to hide title in Docusaurus.
        
    """

    with open(in_html_filepath, "r") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        body = get_content(soup)

    with open(out_md_filepath, "w") as f:
        content = add_docusaurus_metadata(str(body), id, title, hide_title)
        f.write(content)


def get_html(module: str) -> List[str]:
    """Return all html files on the html/module folder"""
    files = glob.glob(f"./html/{module}/*.html")
    # remove ./html/module
    return [f.replace(f"./html/{module}/texthero.", "") for f in files]


def get_prettified_module_name(module_name: str):
    """
    Return a prettified version of the module name. 
    
    Examples
    --------
    >>> get_title("preprocessing")
    Preprocessing
    >>> get_title("nlp")
    NLP
    """
    module_name = module_name.lower().strip()
    if module_name == "nlp":
        return "NLP"
    else:
        return module_name.capitalize()


"""
Update sidebars and markdown files
"""

# make sure folder exists
Path(ROOT_MD_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(ROOT_MD_DIRECTORY + "api").mkdir(parents=True, exist_ok=True)

api_sidebars = {}

for m in MODULES:
    in_html_filename = f"{ROOT_HTML_DIRECTORY}/{m}.html"
    out_md_filename = f"{ROOT_MD_DIRECTORY}/api-{m}.md"
    id = "api-" + m.lower().strip()
    title = get_prettified_module_name(m)

    hide_title = "false"

    # initialize api_sidebars
    api_sidebars[title] = [id]

    to_md(in_html_filename, out_md_filename, id, title, hide_title)


for a in glob.glob("./_build/html/api/*.html"):
    object_name = a.split("/")[-1].replace(".html", "")

    id = object_name
    (_, module_name, fun_name) = object_name.split(".")

    title = f"{module_name}.{fun_name}"

    module_name = get_prettified_module_name(module_name)

    hide_title = "true"

    api_sidebars[module_name].sort()

    api_sidebars[module_name] = api_sidebars[module_name] + ["api/" + id]

    in_html_filename = f"{ROOT_HTML_DIRECTORY}/api/{object_name}.html"
    out_md_filename = f"{ROOT_MD_DIRECTORY}/api/{object_name}.md"

    to_md(in_html_filename, out_md_filename, id, title, hide_title)


# Load, update and save again sidebars.json
with open(SIDEBARS_FILEPATH) as js:
    sidebars = json.load(js)

sidebars["api"] = api_sidebars

with open(SIDEBARS_FILEPATH, "w") as f:
    json.dump(sidebars, f, indent=2)
