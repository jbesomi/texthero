# Contributing

Hi!

Thank you for being here. Texthero is maintained by [jbesomi](https://github.com/jbesomi). He is glad to receive help.

## Getting started

If you feel you want to help and do not know where to start, you may look at  `getting started` [issues](https://github.com/jbesomi/texthero/issues).

## Development workflow

The next steps will guide you towards making contributions on this repository. You just have to follows step-by-step. If anything is not clear or you have an idea on how to improve this document, feel free to edit it and open a pull request.

In case you need a more broad vision, you may need to read through the great [Github Guides](https://guides.github.com/). You can start from [Creating a pull request from a fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) for instance.

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

If may need Sphinx and some [other dependencies](../setup.cfg) to properly update the API docuementation. The following command install for you all the required development-dependencies:

```
pip install -e '.[dev]'
```


1. Create a new working branch

You can name it as you wish. A good practice is to give to the branch a meaningful name so others knows what you are working on.

```
$ git checkout -b new-branch
```

1. Add your changes

Try to commit regularly. Also, when possible, group changes into distinct commits. It will be easier for the rest of us to understand what you worked just by reading the description of your commit.

```
$ ...
```

1. Test your changes

Before opening a new pull-request, you should make sure that all tests still pass with the new changes. 

**Important.** If you worked on a bug, you should add a test that check the bug is not present anymore. This is extremely useful as it avoids to re-introduce the same bug again in the future.

In this part, you need to execute the `check.sh` script. Other than executing all tests, this script will format again all the repository code and [update the documentation](#documentation) with the new changes.


```
cd scripts
./check.sh
```

> To properly execute the check command, you need to make sure you have installed all the required dependencies, in particular Sphinx.

1. Open a Pull Request (PR)

The time to submit the PR has come. Head to your forked repository on Github. Then, switch to your working branch and click on "New pull request". Fill-out the pull request template and then submit it. Someone will review it shortly and eventually get back to you for questions or feedback.


## Scripts folder

- `./check.sh`
   - format the code with yapf (`format.sh`)
   - update the Sphinx documentation for the website
   - Execute all test with `unittest` (`check.sh`)
   - **This is the only and main file that must be called.**
- `./formath.sh`
   - format all code with yapf (soon to be replaced with black)

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

### Style

Terms of style: PEP 8

- Use normal rules for colons, that is, no space before and one space after a colon: text: str.
- Use spaces around the = sign when combining an argument annotation with a default value: align: bool = True.
- Use spaces around the -> arrow: def headline(...) -> str.

### Git commits

- Strive for atomicity: 1 commit = 1 context.
- Write essage in the present tense `Add XYZ support`
- You can reference relevant issues using an hastag plus the number of the issue. Example: `#1`


### Tests

Tests are made with `unittest`.
To execute all test, you can simply
```
$ cd scripts
$ ./tests.sh
```

Calling `./test.sh` is equivalent to execute form the _root_ `python3 -m unittest discover -s tests -t .`
