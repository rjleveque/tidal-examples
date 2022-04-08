
"""
Calculate and plot the tide at Wake Island over 4 weeks and then over 4 days
for discussion in a notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import datetime

#from clawpack.geoclaw import tidetools

# Eventually move tidetools.py to geoclaw
# For development purposes, this is temporarily in this repo:

import os, sys
pathstr = os.path.abspath('..')
if pathstr not in sys.path:
    sys.path.insert(0,pathstr)
import tidetools

 
#Station Information
station_id = '1890000'
datum = 'MTL'

#Beginning and End Dates 
beg_date = datetime.datetime(2022, 2, 25, hour=0)
end_date = datetime.datetime(2022, 4, 3, hour=0)

#Predict tide with arguments set as: (station_id, beg_prediction_date, end_prediction_date)
predicted_tide = tidetools.predict_tide(station_id, beg_date, end_date, datum=datum)

total_hrs=((end_date - beg_date).total_seconds())/3600.
hours = np.arange(0, total_hrs+0.001, 0.1)

def make_tide_plot_4weeks():
    plt.figure(figsize=(13,6))
    plt.plot(hours/24, predicted_tide, 'b')
    plt.xticks(np.arange(0,total_hrs/24,7))
    plt.xlabel('Days since ' + str(beg_date) + '(GMT)')
    plt.ylabel('Meters above MTL')
    plt.xlim(0,29)
    plt.ylim(-0.6,0.6)
    #plt.legend(loc='upper left')
    plt.grid(True)
    plt.title('4 weeks of tide at Wake Island');

def make_tide_plot_4days():
    plt.figure(figsize=(13,6))
    plt.plot(hours, predicted_tide, 'b')
    plt.xticks(np.arange(0,97,12))
    plt.xlabel('Hours since ' + str(beg_date) + '(GMT)')
    plt.ylabel('Meters above MTL')
    plt.xlim(0,96)
    plt.ylim(-0.6,0.6)
    #plt.legend(loc='upper left')
    plt.grid(True)
    plt.title('4 days of tide at Wake Island');