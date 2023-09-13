import json
import pandas as pd
import time
import requests
import warnings
warnings.filterwarnings("ignore")

# Request headers
headers = {
    'X-ApiKey': "a2889e0c-6f9f-4366-9db1-63eed65b7564"
}

delivery_areas = ['FI', 'NO1', 'NO2', 'NO3', 'NO4', 'NO5', 'SE1', 'SE2', 'SE3', 'SE4', 'DK1', 'DK2']
regulation_price_areas = ['SE1, Up', 'SE2, Up', 'SE3, Up', 'SE4, Up',
                          'NO1, Up', 'NO2, Up', 'NO3, Up', 'NO4, Up', 'NO5, Up',
                          'DK1, Up', 'DK2, Up', 'FI, Up',
                          'SE1, Down', 'SE2, Down', 'SE3, Down', 'SE4, Down',
                          'NO1, Down', 'NO2, Down', 'NO3, Down', 'NO4, Down', 'NO5, Down',
                          'DK1, Down', 'DK2, Down', 'FI, Down']
consumption_prognosis_areas = ['FI', 'NO', 'SE', 'DK1', 'DK2']
dk_areas = ['DK1', 'DK2']
se_areas = ['SE1', 'SE2', 'SE3', 'SE4']
no_areas = ['NO1', 'NO2', 'NO3', 'NO4', 'NO5']
hydro_areas = ['FI', 'NO', 'SE']

baseURL = r'https://api.helen.fi/api/if034/Nordpool/{}?{}'

old_nordpool_data = pd.read_csv("data/nordpool_data_filtered.csv", sep='|', index_col=[0])
old_nordpool_data.index = pd.to_datetime(old_nordpool_data.index)

with open(r'data_mappings\nordpool_dev.json') as series_mapping:
    mapping_data = json.load(series_mapping)


def fill_series(mapping, area, old_nordpool_data):
    if not area:
        series_name = mapping['name']
    else:
        series_name = mapping['name'] + ', ' + area
    series = old_nordpool_data[[series_name]]
    series.dropna(inplace=True)
    startTime = series.index.max()
    parameters = 'startTime={}'.format(startTime.strftime('%Y-%m-%dT%H:00:00Z'))
    if 'EUR' in mapping['name']:
        parameters += "&currency=EUR"
    if area:
        parameters += "&deliveryarea=" + area.split(',')[0]
    if mapping['name'] == "Saatohinta, EUR/MWh":
        if area.split(',')[-1] == ' Down':
            endpoint = mapping["endpoint"] + 'down'
        else:
            endpoint = mapping["endpoint"] + 'up'
    else:
        endpoint = mapping["endpoint"]
    url = baseURL.format(endpoint, parameters)
    response = requests.get(url=url, headers=headers)
    response_json = json.loads(response.content.decode('utf-8'))
    reponse_df = pd.DataFrame(response_json['Body'][0]['values'], columns=['startTime', 'value'])
    reponse_df.rename(columns={'value': series_name, 'startTime': 'DateTime'}, inplace=True)
    reponse_df.set_index('DateTime', inplace=True, drop=True)
    reponse_df.index = pd.to_datetime(reponse_df.index)
    if 'Vesi' in series_name:
        reponse_df = reponse_df.resample('1h').interpolate('linear')
    series = pd.concat([series, reponse_df])
    series = series[~series.index.duplicated(keep='first')]
    time.sleep(0.5)
    return series


filled_data_list = []
n_series_types = mapping_data.__len__()
i = 1

for mapping in mapping_data:
    if mapping['name'] == "Elspot-hinta, EUR/MWh, SYS":
        filled_data = fill_series(mapping, None, old_nordpool_data)
        filled_data_list.append(filled_data)
    elif mapping['name'] == "Vesivoimareservi, GWh":
        for hydro_area in hydro_areas:
            filled_data = fill_series(mapping, hydro_area, old_nordpool_data)
            filled_data_list.append(filled_data)
    elif mapping['name'] == "Kulutusennuste, MWh":
        for consumption_prognosis_area in consumption_prognosis_areas:
            filled_data = fill_series(mapping, consumption_prognosis_area, old_nordpool_data)
            filled_data_list.append(filled_data)
    elif mapping['name'] in ("Tuulituotantoennuste, MWh, DK", "Tuulituotanto, MWh, DK", "Tuotanto, MWh, DK", "Kulutus, MWh, DK"):
        for dk_area in dk_areas:
            filled_data = fill_series(mapping, dk_area, old_nordpool_data)
            filled_data_list.append(filled_data)
    elif mapping['name'] in ("Tuulituotantoennuste, MWh, SE", "Tuulituotanto, MWh, SE", "Tuotanto, MWh, SE", "Kulutus, MWh, SE"):
        for se_area in se_areas:
            filled_data = fill_series(mapping, se_area, old_nordpool_data)
            filled_data_list.append(filled_data)
    elif mapping['name'] in ("Tuotanto, MWh, NO", "Kulutus, MWh, NO"):
        for no_area in no_areas:
            filled_data = fill_series(mapping, no_area, old_nordpool_data)
            filled_data_list.append(filled_data)
    elif mapping['name'] == "Saatohinta, EUR/MWh":
        for regulation_price_area in regulation_price_areas:
            filled_data = fill_series(mapping, regulation_price_area, old_nordpool_data)
            filled_data_list.append(filled_data)
    else:
        for delivery_area in delivery_areas:
            filled_data = fill_series(mapping, delivery_area, old_nordpool_data)
            filled_data_list.append(filled_data)
    print('{} % done'.format(int(i / n_series_types * 100)))
    i += 1

filled_data = filled_data_list[0]

for filled_d_left in filled_data_list[1:]:
    filled_data = filled_data.join(filled_d_left, how='outer')

filled_data.to_csv("data/nordpool_data_filtered.csv", index=True, sep='|')