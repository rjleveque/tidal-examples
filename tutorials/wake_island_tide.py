
"""
Calculate and plot the tide at Wake Island over 4 weeks and then over 4 days
for discussion in a notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
import datetime
import os, sys

 
#Station Information
station_id = '1890000'
datum = 'MTL'

#Beginning and End Dates 
beg_date = datetime.datetime(2022, 2, 25, hour=0)
end_date = datetime.datetime(2022, 4, 3, hour=0)

use_stored_data = True
fname_data = 'wake_island_tide_data.txt'

if use_stored_data:
    hours, predicted_tide = np.loadtxt(fname_data, unpack=True)
    total_hrs = hours[-1]
    print('Loaded data from %s' % fname_data)

else:
    # fetch the data and store it for later use:

    #from clawpack.geoclaw import tidetools

    # Eventually move tidetools.py to geoclaw
    # For development purposes, this is temporarily in this repo:

    CLAW = os.environ['CLAW']
    pathstr = os.path.join(CLAW,'tidal-examples')
    if pathstr not in sys.path:
        sys.path.insert(0,pathstr)
    import tidetools

    predicted_tide = tidetools.predict_tide(station_id, beg_date, 
                                            end_date, datum=datum)

    total_hrs=((end_date - beg_date).total_seconds())/3600.
    hours = np.arange(0, total_hrs+0.001, 0.1)

    data = np.vstack((hours, predicted_tide)).T
    np.savetxt(fname_data, data)
    print('Saved %s' % fname_data)


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
