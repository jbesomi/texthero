from setuptools import find_packages, setup
import os
import codecs



def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

long_description = read('README.md')

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="texthero",
    version=get_version("texthero/__init__.py"),
    description="Text preprocessing, representation and visualization from zero to hero.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Jonathan Besomi",
    project_urls={
        "Documentation": "https://texthero.org",
        "Source": "https://github.com/jbesomi/texthero"
        #"Changelog": "",
    },
    url="https://github.com/jbesomi/texthero",
    keywords = ['text analytics'],
    install_requires=install_requires,
    license="MIT",
    zip_safe=False,
    packages=find_packages(),
    classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3'
  ]
)
