# Tide Prediction

This directory contains code to predict tide for NOAA gauge stations when given a station ID and date(s) of prediction.

Method used to predict tides in this example is adapated from Pytides, a python package developed by Sam Cox. The method used is that of harmonic constituents, in particular as presented by P. Schureman in Special Publication 98.

Harmonic Constituents data is scraped from NOAA's gauge station harmonic constituents page. 

# Requirements:

* Scipy
* Pandas 
* Request 
* lxml

To install dependencies, run <b> pip install -r requirements.txt </b> . 

# Usage
Use [Boundary_Conditions.ipynb](Boundary_Conditions.ipynb) to modify tide prediction examples, develop your own tide predictions, and obtain code for Clawpack implementation if placed in <b>gauge_afteraxes( )</b> in <b>setplot.py</b> and calling <b>surge( )</b> method with arguments set as: <b>(stationID, beginning_date, end_date, landfall_date)</b> to obtain NOAA's observed storm surge.

Modify NOAA station ID and date(s) of prediction (UTC) to make your own example. 

# Acknowledgments
This code was modified from [Pytides](https://github.com/sam-cox/pytides) to work with Python 3 and to more easily predict NOAA gauge station tides and storm surges with only calling <b>surge( )</b> method with a stationID, a beginning date, an end date, and a landfall date in setplot.py to compare Clawpack storm surge ouput to NOAA's observed storm surge. 
