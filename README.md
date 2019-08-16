# pypath
pypath is a passive vehicle tracking attack

## Setup
1. Install **python3** ([OSX](http://docs.python-guide.org/en/latest/starting/install/osx/), [Linux](http://docs.python-guide.org/en/latest/starting/install/linux/), [Windows](https://docs.continuum.io/anaconda/install))
2. Create a virtual environment
* pyenv </path/to/yourenvname> (OSX, Linux)
* conda create -n </path/to/yourenvname> python=x.x anaconda (Windows)
3. Install requirements
* </path/to/yourenvname>/bin/pip3 install -r </path/to/repository/>requirements.txt (OSX, Linux)
* </path/to/yourenvname>/bin/conda install --yes --file </path/to/repository/>requirements.txt (Windows)

## Running
main.py currently accepts a single command line argument *track*. *track* accepts optional arguments:
* *s* tracking strategy that to be used (default=VET)
* *n* number of subjects to consider (default=all)

