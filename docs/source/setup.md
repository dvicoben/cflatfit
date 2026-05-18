# Set-up

C♭Fit is python-based, requiring python >= 3.11. There is currently no `pyPI` installation, but C♭Fit can be easily installed after cloning the repository.


## Requirements
Requirements are listed in [`requirements.txt`](https://github.com/dvicoben/cflatfit/blob/master/requirements.txt). Uses standard libraries (`numpy`, `matplotlib`, `scipy`), and `iminuit`.

## Using pip
In the python environment of your choice, 

```
git clone https://github.com/dvicoben/cflatfit.git
cd cflatfit
pip install -e .
```

## Quick
Ensure you are in a python environment with all the requirements in [`requirements.txt`](https://github.com/dvicoben/cflatfit/blob/master/requirements.txt)
```
git clone https://github.com/dvicoben/cflatfit.git
cd cflatfit
pip install -e .
```
The script setup.sh simply appends `src/` to the `PYTHONPATH` so that contents therein will be found when running scripts. You will need to `source ./setup.sh` whenever you start a new terminal session.