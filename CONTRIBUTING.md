# CONTRIBUTING

Hello and welcome to Texthero!

This document contains all the important information you need to get started contributing.


## Vision

In case you are interested in the Texthero's vision as well as the core-principle, have a look at [PURPOSE.md](./PURPOSE.md)


## Quality

Texthero's main goal is to make the NLP-developer life _easier_. It does so by
1. Providing a simple-yet-complete tool for NLP and text analytics
2. Empowering the NLP developer with great documentation, simple getting started docs as well as (work in progress) clear and concise tutorials (blog).

To achieve all of this, Texthero's code and documentation must be of high quality. Having a clean, readable, and **tested** code drastically reduces the likelihood of introducing bugs, and having great documentation will facilitate the work of many NLP developers as well as the work of Texthero's maintainers.


## Shift-left testing

Texthero follows an approach known as shift-left testing. According to [Wikipedia](https://en.wikipedia.org/wiki/Shift-left_testing):

> Shift-left testing is an approach to software testing and system testing in which testing is performed earlier in the lifecycle.

Shift-left testing reduces the number of bugs by attempting to solve the problem at the origin. Often many programming defects are not uncovered and fixed until after significant effort has been wasted on their implementation. Texthero attempts to avoid these kind of issues.


## Improve documentation!

A very important yet not particularly complex task consists in improving the documentation: many Texthero's users will be deeply grateful for your effort.

For instance, as of now, [texthero.representation.nmf](https://texthero.org/docs/api/texthero.representation.nmf) is very poor.

> Interested in improving this? It's pretty easy. Just copy-paste the docstring from texthero.representation.nmf and replace 'pca' with 'nmf' :D


## How to create a successful Pull Request on Texthero

Making sure your pull requests do not break the code and bring something valuable to the project means that only _high quality_ pull requests are approved.

The following link gives some advice on how to submit a successful pull request.

1. Submitting a successful PR is not hard. Have a look at all [previous PR](https://github.com/jbesomi/texthero/pulls?q=is%3Apr+is%3Aclosed) already approved.
1. **Extensively test your code**. Think at all possible edge cases. Look at similar tests for ideas.
1. In most cases, there exists an example of function or docstring very similar to your specific use-case. Before writing your own-code, look at what the other functions look like.
1. Before submitting, **test locally** that you pass all tests (see below under `testing`).
1. Respect the best practices (see below `best practices`).
1. Make sure your code is black-formatted (`./format.sh`, see `formatting`).

<!--
1. Make use of the PR template (see `PR template` ) -->


## Ask questions!

We are there for you! If anything is unclear, just ask. We will do our best to answer you quickly.

## Propose new ideas!

Texthero is there for the NLP-community. If you have an idea on how we can improve it, let us know by opening a new [issues](https://github.com/jbesomi/texthero/issues). We will be glad to hear from you!

## Best practices

1. Make sure Pull Request only changes one thing and one thing only. PR should be independent and self-contained. Read this article: [A Plea For Small Pull Requests](https://opensource.zalando.com/blog/2017/10/small-pull-requests/)
1. Name your PR title accordingly to your changes and add a good and exhaustive description
1. Give the branch a meaningful name. Avoid using the master branch.
1. Maximum line length (for lines of code) should be 88 characters (default settings of `black`).

### Best practices for docstrings

1. Make sure you read and you respect the [numpydoc docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html) and the [PEP 257 -- Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
1. Before writing a new function or make any changes, look at similar code for inspiration and to learn about the code format and style. 
1. The maximal docstring line length should be 75 characters. This should be manually done as `black` formatting does not enforce limits on docstring line length.
1. Use American English instead of British English (e.g. categorize instead of categorise) when writing comments and documenting docstrings.
1. For default argument values, use the defaults from the underlying library if applicable (e.g. the default arguments
from sklearn if using a sklearn algorithm). If other values are used, add a small comment explaining why. Additionally, look for similar functions and use their default values.
1. Default values are defined as follows: `x : int, optional, default=2`
1. In docstring examples, on long `pipe`, consider enclosing the code in parenthesis like so:
    ```
    >>> s = (
    ...     s.pipe(hero.clean)
    ...      .pipe(hero.tokenize)
    ...      .pipe(hero.term_frequency)
    ...      .pipe(hero.flatten)
    ... )
    ```
1. This is an example of a correct defined docstring:


```python
def remove_digits(input: pd.Series, only_blocks=True) -> pd.Series:
   """
   Remove all digits from a series and replace it with a single space.

   Parameters
   ----------

   input : pd.Series
   only_blocks : bool, optional, default=True
               Remove only blocks of digits. For instance, `hel1234lo 1234` becomes `hel1234lo`.

   Examples
   --------
   >>> s = pd.Series("7ex7hero is fun 1111")
   >>> remove_digits(s)
   0    7ex7hero is fun 
   dtype: object
   >>> remove_digits(s, only_blocks=False)
   0    exhero is fun 
   dtype: object
   """
```

## Good first issue

If this is your first time contributing to Texthero, you might start by choosing a `good first issue` [issues](https://github.com/jbesomi/texthero/issues).


## Testing

As you understood, Texthero is serious about testing. We strongly encourage contributors to embrace [test-driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development).

Tests are made with `unittest` from the python standard library: [Unit testing framework](https://docs.python.org/3/library/unittest.html)
 
To execute all tests, you can simply
```
$ cd scripts
$ ./tests.sh
```

Calling `./tests.sh` is equivalent to executing it from the _root_ `python3 -m unittest discover -s tests -t .`


**Important.** If you worked on a bug, you should add a test that checks the bug is not present anymore. This is extremely useful as it avoids to re-introduce the same bug again in the future.


### Passing doctests

When executing `./tests.sh` it will also check that the Examples in the docstrings are correct (doctests).

Passing doctests might be a bit annoying sometimes. Let's look at this example for instance:

```
File "/home/travis/build/jbesomi/texthero/texthero/preprocessing.py", line 700, in texthero.preprocessing.remove_tags
Failed example:
    hero.remove_tags(s)
Expected:
    0    instagram texthero
    dtype: object 
Got:
    0    instagram texthero
    dtype: object
```

The docstring failed? Why? The reason is that somewhere in the `Example` section of docstring, we missed one or more white spaces ` `.

### Travis CI

When you submit your code, all code will be tested on different operating systems using Travis CI: [TRAVIS CI texthero](https://travis-ci.com/github/jbesomi/texthero).

Make sure you pass all your tests locally before opening a pull request!

## Formatting

Before submitting, make sure your code is formatted. Code formatting is done with [black](https://github.com/psf/black).

```
cd scripts
./format.sh
```

Travis CI will check that the whole code is black-formatted. Make sure you format before submitting!

> It's handy to install the black formatter directly on your IDE.


## Development workflow

In case you need a more broad vision on how contributions work on Github, please refers to the [Github Guides](https://guides.github.com/). For getting started, you might find [Creating a pull request from a fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) useful.

1. Fork the repository

1. Clone the repository

1. Connect your cloned repository to the _original_ repo

```
$ cd texthero
$ git remote add upstream git@github.com:jbesomi/texthero.git
```

> This first step needs to be done only once. But, in the future when you will want to make new changes, make sure your repository is synchronized with respect to the upstream: [Syncing a fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork).

1. Install texthero locally **and** his dev-dependencies

Install texthero locally directly from the source code. This makes sure you are working on the latest version and that you will install all the required dependencies.

```
$ pip install -e .
```
> The `-e` will install the python package in 'development' mode. That way your changes will take effect immediately without the need to reinstall the package again.

```
pip install -e '.[dev]'
```

- All contributors are expected to install the development dependencies as well.

- Some of the [dev-dependencies](https://github.com/jbesomi/texthero/blob/6e6b8f70432979a81a09d48826fc907adc67cba7/setup.cfg#L43) will be used by any contributor of TextHero to execute the [tests.sh](./scripts/tests.sh) locally.

- execute `pre-commit install` inside your project folder in order to enable git pre commit hook. This will format your code automatically before staging them

## **IMPORTANT NOTE**

- Some of the [dev-dependencies](https://github.com/jbesomi/texthero/blob/6e6b8f70432979a81a09d48826fc907adc67cba7/setup.cfg#L43) are necessary **IF** the contributor wants to update the website or run the website locally but please remember that one **shouldn't be sending these kind of changes as a Pull Request**.

**Why?**

- Because it would instantaneously change the website.
- Changes from pull requests will be available to everyone only after a new release. Imagine you add a new function as a PR, if your PR also updates the documentation then the function will appear under the APIs which is not yet present in the installable pip version. That would be really confusing, isn't it?


1. Create a new working branch

You can name it as you wish. A good practice is to give the branch a meaningful name so others know what you are working on.

**Example branch name**: `33-fixing-wordcloud-issue`. Here `33` indicates this [issue tracker ID](https://github.com/jbesomi/texthero/issues/33), `fixing-wordcloud-issue` is a short and actionable description of what your PR is about. Use hyphens as separators. 

```
$ git checkout -b new-branch
```

1. Add your changes

Try to commit regularly. In addition, whenever possible, group changes into distinct commits. It will be easier for the rest of us to understand what you worked on just by reading the description of your commit.

```
$ git add README.md
$ git commit -m "added README.md"
```

1. Test your changes

Before opening a new pull-request, you should make sure that all tests still pass with the new changes. Also, if you implement a new function or enhance an existing one, please **add all the necessary** unittests. PR without a properly unit-tested code will not be accepted as we want to avoid bugs at all costs in the project. This is also known as [Shift-left testing](https://en.wikipedia.org/wiki/Shift-left_testing).

**Important.** If you worked on a bug, you should add a test that checks the bug is not present anymore. This is extremely useful as it avoids to re-introduce the same bug again in the future.


1. Open a Pull Request (PR)

The time to submit the PR has come. Head to your forked repository on Github. Then, switch to your working branch and click on "New pull request". Fill-out the pull request template and then submit it. We will review it shortly and eventually get back to you for questions or feedback.

## Scripts folder

- `./test.sh`
   - Execute unittests as well as test all doctests
- `./format.sh`
   - format all code with [black](https://github.com/psf/black)
- `./check.sh`
   - Format the code with black (`format.sh`)
   - Update the Sphinx documentation for the website
   - Execute all test with `unittest` (`check.sh`)


### Git commits

- Strive for atomicity: 1 commit = 1 context.
- Write messages in the present tense `Add XYZ support`
- You can reference relevant issues using a hashtag plus the number of the issue. Example: `#1`


**Work in progress:** this document is a work in progress. If you spot a mistake or you want to make something clear, open a pull request!
