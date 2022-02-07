import datetime
import requests
import lxml.html as lh
import pandas as pd
import json

from tide import Tide
import constituent as cons
import numpy as np

def retrieve_constituents(stationID):
    #Forms URL
    url = 'https://tidesandcurrents.noaa.gov/harcon.html?unit=0&timezone=0&id={}'.format(stationID)

    #Requests URL 
    page = requests.get(url)
    doc = lh.fromstring(page.content)
    tr_elements = doc.xpath('//tr')
    col = [((t.text_content(),[])) for t in tr_elements[0]]
    for j in range(1, len(tr_elements)):
        T, i = tr_elements[j], 0
        for t in T.iterchildren():
            col[i][1].append(t.text_content())
            i+=1
 
    #Appends data to csv file
    component_dict = {title:column for (title,column) in col}
    component_array = pd.DataFrame(component_dict)
    component_array.to_csv('{}_constituents.csv'.format(stationID), index=False)
    
    return component_dict


def retrieve_water_levels(*args):
    #NOAA api
    api = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={}'\
          '&end_date={}&'.format(args[1].strftime("%Y%m%d %H:%M"), args[2].strftime("%Y%m%d %H:%M"))
    
    #NOAA observed data 
    obs_url = 'station={}&product=water_level&datum=MTL&units=metric&time_zone=gmt'\
              '&application=ports_screen&format=json'.format(args[0])

    obs_data_page = requests.get(api + obs_url)
    obs_data = obs_data_page.json()['data']
    obs_heights = [float(d['v']) for d in obs_data]
    
    NOAA_times = [datetime.datetime.strptime(d['t'], '%Y-%m-%d %H:%M') for d in obs_data]
        
    component_dict = {'Datetimes': NOAA_times, 'Observed Heights': obs_heights}
    component_array = pd.DataFrame(component_dict)
    component_array.to_csv('{}_NOAA_water_levels.csv'.format(args[0]), index=False)
    
    return NOAA_times, obs_heights


def retrieve_predicted_tide(*args):
    #NOAA api
    api = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={}'\
          '&end_date={}&'.format(args[1].strftime("%Y%m%d %H:%M"), args[2].strftime("%Y%m%d %H:%M"))
    
    #NOAA predicted data 
    pred_url = 'station={}&product=predictions&datum=MTL&units=metric&time_zone=gmt'\
               '&application=ports_screen&format=json'.format(args[0])
    
    pred_data_page = requests.get(api + pred_url)
    pred_data = pred_data_page.json()['predictions']
    pred_heights = [float(d['v']) for d in pred_data]
    
    return pred_heights
    
    
def datum_value(stationID, datum): 
    #Scrapes MTL/MSL Datum Value
    datum_url = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?&station={}&product=datums'\
                '&units=metric&time_zone=gmt&application=ports_screen&format=json'.format(stationID)
    page_data = requests.get(datum_url)
    data = page_data.json()['datums']       
    datum_value = [d['v'] for d in data if d['n'] == datum]
    
    return float(datum_value[0])

     
def predict_tide(*args): 
    #These are the NOAA constituents, in the order presented on their website.
    constituents = [c for c in cons.noaa if c != cons._Z0]
    noaa_values = retrieve_constituents(args[0])
    noaa_amplitudes = [float(amplitude) for amplitude in noaa_values['Amplitude']]
    noaa_phases = [float(phases) for phases in noaa_values['Phase']] 
    
    #We can add a constant offset (e.g. for a different datum, we will use relative to MLLW):
    MTL = datum_value(args[0], 'MTL')
    MSL = datum_value(args[0], 'MSL')
    offset = MSL - MTL
    constituents.append(cons._Z0)
    noaa_phases.append(0)
    noaa_amplitudes.append(offset)
       
    #Build the model
    assert(len(constituents) == len(noaa_phases) == len(noaa_amplitudes))
    model = np.zeros(len(constituents), dtype = Tide.dtype)
    model['constituent'] = constituents
    model['amplitude'] = noaa_amplitudes
    model['phase'] = noaa_phases
    tide = Tide(model = model, radians = False)
    
    #Time Calculations
    delta = (args[2]-args[1])/datetime.timedelta(hours=1) + .1
    times = Tide._times(args[1], np.arange(0, delta, .1))
    
    #Height Calculations
    heights_arrays = [tide.at([times[i]]) for i in range(len(times))]
    pytide_heights = [val for sublist in heights_arrays for val in sublist]
 
    return pytide_heights 


def datetimes(*args):
    #Time Calculations 
    delta = (args[1]-args[0])/datetime.timedelta(hours=1) + .1
    times = Tide._times(args[1], np.arange(0, delta, .1))
    
    return times


def detide(*args):
    # NOAA observed water level - predicted tide 
    return [(args[0][i] - args[1][i]) for i in range(len(args[0]))] 

    
def surge(*args):
    #Surge Implementation
    predicted_tide = predict_tide(args[0], args[1], args[2])
    NOAA_times, NOAA_observed_water_level = retrieve_water_levels(args[0], args[1], args[2])
    surge = detide(NOAA_observed_water_level, predicted_tide) 
    times = [(time-args[3])/datetime.timedelta(days=1) for time in NOAA_times]
    
    return times, surge

