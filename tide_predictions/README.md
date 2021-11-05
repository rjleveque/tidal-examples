# Tide Prediction

This directory contains code to predict tide for NOAA gauge stations when given a station ID and date(s) of prediction.

Method used to predict tides in this example is adapated from Pytides, a python package developed by Sam Cox. The method used is that of harmonic constituents, in particular as presented by P. Schureman in Special Publication 98.

Harmonic Constituents data is scraped from NOAA gauge station harmonic constituents page. 

# Requirements:

* Numpy 
* Scipy
* Matplotlib
* Pandas 
* Request 
* lxml

# Installations:

```
pip install numpy
pip install scipy
pip install matplotlib
pip install pandas
pip install requests
pip install lxml
```

# Usage
Use [Boundary_Conditions.ipynb](https://github.com/socoyjonathan/tide_predictions/blob/main/Boundary_Conditions.ipynb) to modify tide prediction examples.

Modify NOAA station ID and date(s) of prediction (GMT) to make your own example.

Data for tide predictions utilize meters and radians.

# Acknowledgments
This code was modified from [Pytides](https://github.com/sam-cox/pytides) to work with Python 3 and to more easily predict NOAA gauge station tides. 

For questions about the code in this example, email js5587@columbia.edu




