import requests
import lxml.html as lh
import pandas as pd

def scrapeURL(stationID):
    
    #Forms URL
    url = "https://tidesandcurrents.noaa.gov/harcon.html?unit=0&timezone=0&id={}".format(stationID)

    #Requests URL 
    page = requests.get(url)
    doc = lh.fromstring(page.content)
    tr_elements, col = doc.xpath('//tr'), []
    
    #Appends numerical data to each constituent
    for t in tr_elements[0]:
        name = t.text_content()
        col.append((name,[]))
        
    for j in range(1, len(tr_elements)):
        T, i = tr_elements[j], 0
        for t in T.iterchildren():
            data = t.text_content()
            col[i][1].append(data)
            i+=1
            
    #Appends data to csv file
    component_dict = {title:column for (title,column) in col}
    component_array = pd.DataFrame(component_dict)
    component_array.to_csv('{}_constituents.csv'.format(stationID), index=False)