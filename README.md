# C♭Fit (Custom Flexible Arbitrary Template Fit)
C♭Fit (cflatfit) is a small package for template fits. 

This package is designed mainly for personal use, and it is therefore subject to rapid change depending on my particular use-cases. It should not be regarded as stable.


# Requirements
Python >= 3.11, and packages listed in `requirements.txt`. Uses standard libraries (`numpy`, `matplotlib`), and `iminuit`.


# Set-up
## Quick
Ensure you are in a python environment with all requirements in `requirements.txt`, then
```
git clone https://github.com/dvicoben/cflatfit.git
cd cflatfit
source ./setup.sh
```
The script `setup.sh` simply appends `src/` to the `PYTHONPATH` so that contents therein will be found when running scripts. You will need to `source ./setup.sh` whenever you start a new terminal session.

## Using pip
In the python environment of your choice, 
```
git clone https://github.com/dvicoben/cflatfit.git
cd cflatfit
pip install -e .
```
which should install the package (`cflatfit`) and the required dependencies.

# TO DO:
- [ ] Incorporate PDFBase external constraint to minimizaiton (e.g. for sum of yields)
