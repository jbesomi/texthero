# Contributing

Hello there!

Thank you for being here. Texthero is maintained by [jbesomi](https://github.com/jbesomi). He is glad to receive your help.

## Getting started

If you feel you want to help and do not know where to start, you may start with the `good first issue` [issues](https://github.com/jbesomi/texthero/issues).

## Development workflow

The next steps will guide you towards making contributions to this repository. You just have to follows step-by-step. If anything is not clear or you have an idea on how to improve this document, feel free to edit it and open a pull request.

In case you need a more broad vision on how contributions work on Github, please refers to the [Github Guides](https://guides.github.com/). For getting started, read also [Creating a pull request from a fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork).

If you are used to the Github workflow, you can find at the end of this document a summary of the most important parts.

1. Fork the repository
   Click the `fork` button in the GitHub repository; this will create a copy of Texthero in your Github account.

1. Clone the repository
   To do that, you need to have [git](https://git-scm.com/) installed. Open the terminal and type

```
$ git clone git@github.com:YOUR_USERNAME/texthero.git
```
1. Connect your cloned repository to the _original_ repo

```
$ cd texthero
$ git remote add upstream git@github.com:jbesomi/texthero.git
```

> This first step needs to be done only once. If in the future you will want to make new changes, make sure your repository is synchronized with respect to the upstream: [Syncing a fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork).

1. Install texthero locally and his dev-dependencies

Install texthero locally directly from the source code. This makes sure you are working on the latest version and that you will install all the required dependencies.

```
$ pip install -e .
```

> The `-e` will install the python package in 'development' mode. That way your changes will take effect immediately without the need to reinstall the package again.

1. Install development dependencies

Development dependencies need to be installed to update the website documentation, i.e the content in texthero.org. 

In most cases, you **do not need** to update this. Changes from pull requests will be available to everyone only after a new release.

```
pip install -e '.[dev]'
```


1. Create a new working branch

You can name it as you wish. A good practice is to give the branch a meaningful name so others know what you are working on.

```
$ git checkout -b new-branch
```

1. Add your changes

Try to commit regularly. In addition, whenever possible, group changes into distinct commits. It will be easier for the rest of us to understand what you worked just by reading the description of your commit.

```
$ ...
```

1. Test your changes

Before opening a new pull-request, you should make sure that all tests still pass with the new changes. Also, if you implement a new function or enhance an existing one, please **add all the necessary** unittests. PR without a properly unit-tested code will not be accepted as we want to avoid at all costs bugs in the project.

**Important.** If you worked on a bug, you should add a test that checks the bug is not present anymore. This is extremely useful as it avoids to re-introduce the same bug again in the future.

In this part, you need to execute:
 - `./format.sh` that will format all code with `black`
- `./test.sh` that will test all unittests and doctests. 

> In the scripts folder there is also a `check.sh` shell script. Other than executing all tests, `check.sh` script will format again all the repository code and [update the documentation](#documentation) with the new changes. In most cases, you don't need to execute this one. To properly execute the check command, you need to make sure you have installed all the required dependencies, in particular Sphinx.

```
cd scripts
./format.sh
./test.sh
```

1. Open a Pull Request (PR)

The time to submit the PR has come. Head to your forked repository on Github. Then, switch to your working branch and click on "New pull request". Fill-out the pull request template and then submit it. We will review it shortly and eventually get back to you for questions or feedback.

## Scripts folder

- `./test.sh`
   - Execute unittests as well as test all doctests
- `./formath.sh`
   - format all code with [black](https://github.com/psf/black)
- `./check.sh`
   - format the code with black (`format.sh`)
   - update the Sphinx documentation for the website
   - Execute all test with `unittest` (`check.sh`)
   - **This is the only and main file that must be called.**

## Good to know

1. Passing doctests might be a bit annoying sometimes. Let's look at this example for instance:

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

The docstring failed but it's not particularly clear why, right? Here, the reason is that somewhere on the docstring `Example`, we missed one or more white spaces ` `.

## Conventions

### Documentation and website

Texthero docstring follows [NumPy/SciPy](https://numpydoc.readthedocs.io/en/latest/format.html) docstring style. For example:

```python
def remove_digits(input: pd.Series, only_blocks=True) -> pd.Series:
   """
   Remove all digits from a series and replace it with a single space.

   Parameters
   ----------

   input : pd.Series
   only_blocks : bool
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
   ...
```


### Git commits

- Strive for atomicity: 1 commit = 1 context.
- Write messages in the present tense `Add XYZ support`
- You can reference relevant issues using a hashtag plus the number of the issue. Example: `#1`


## Test-driven development

Texthero is serious about testing. We strongly encourage contributors to embrace [test-driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development).

Tests are made with `unittest` from the python standard library: [Unit testing framework](https://docs.python.org/3/library/unittest.html)
 
To execute all tests, you can simply
```
$ cd scripts
$ ./tests.sh
```

Calling `./test.sh` is equivalent to execute form the _root_ `python3 -m unittest discover -s tests -t .`
