import io
import json
import sqlite3
import time
import pandas as pd
import requests

""" This script collects Fingrid open data as is. Documentation https://data.fingrid.fi/pages/api """

with open(r'keys/fingrid_keys_dev.json') as api_keys:
    x_api_keys = json.load(api_keys)

# Request headers
headers = {
    'Accept': 'text/csv',
    'x-api-key': x_api_keys["key"],
}

# Requests made for a given year
update_year = 2023

# Start and end data for splitting requests for higher frequencies (1, 3, 15 min) based on the API per request row limit
star_end_dates = {
    '1': [(r'-01-01T00:00:00Z', r'-02-01T00:00:00Z'), (r'-02-01T00:00:01Z', r'-03-01T00:00:00Z'),
          (r'-03-01T00:00:01Z', r'-04-01T00:00:00Z'), (r'-04-01T00:00:01Z', r'-05-01T00:00:00Z'),
          (r'-05-01T00:00:01Z', r'-06-01T00:00:00Z'), (r'-06-01T00:00:01Z', r'-07-01T00:00:00Z'),
          (r'-07-01T00:00:01Z', r'-08-01T00:00:00Z'), (r'-08-01T00:00:01Z', r'-09-01T00:00:00Z'),
          (r'-09-01T00:00:01Z', r'-10-01T00:00:00Z'), (r'-10-01T00:00:01Z', r'-11-01T00:00:00Z'),
          (r'-11-01T00:00:01Z', r'-12-01T00:00:00Z'), (r'-12-01T00:00:01Z', r'-12-31T23:59:00Z')],
    '3': [(r'-01-01T00:00:00Z', r'-03-01T00:00:00Z'), (r'-03-01T00:00:01Z', r'-05-01T00:00:00Z'),
              (r'-05-01T00:00:01Z', r'-07-01T00:00:00Z'), (r'-07-01T00:00:01Z', r'-09-01T00:00:00Z'),
              (r'-09-01T00:00:01Z', r'-11-01T00:00:00Z'), (r'-11-01T00:00:01Z', r'-12-31T23:59:00Z')],
    '5': [(r'-01-01T00:00:00Z', r'-03-01T00:00:00Z'), (r'-03-01T00:00:01Z', r'-05-01T00:00:00Z'),
          (r'-05-01T00:00:01Z', r'-07-01T00:00:00Z'), (r'-07-01T00:00:01Z', r'-09-01T00:00:00Z'),
          (r'-09-01T00:00:01Z', r'-11-01T00:00:00Z'), (r'-11-01T00:00:01Z', r'-12-31T23:59:00Z')],
    '15': [(r'-01-01T00:00:00Z', r'-07-01T00:00:00Z'), (r'-07-01T00:00:01Z', r'-12-31T23:59:00Z')]
}


def request_data(start_time, end_time, year, variableId):
    """ Make a single request with start time and end time """
    params = (
        ('start_time', str(year) + start_time),
        ('end_time', str(year) + end_time),
    )

    try:
        response = requests.get(r'https://api.fingrid.fi/v1/variable/' + str(variableId) + r'/events/csv',
                                headers=headers, params=params, verify=True)
    except:
        print('Data request failed for {}!'.format(variableId))
        return pd.DataFrame()

    response_csv = io.StringIO(response.content.decode('utf-8'))
    series_data = pd.read_csv(response_csv)
    series_data['DateTime'] = pd.to_datetime(series_data['start_time'])
    series_data = series_data.set_index('DateTime')
    series_data = series_data.drop(columns=['end_time', 'start_time'])
    return series_data


def get_full_year(year, variableId):
    """ Returns a single year of data for a specified series. Splits the requests based on time frequency of data """

    variableId = str(variableId)
    with open(r'data_mappings/fingrid_fill.json') as series_mapping:  # Get time frequency mapping for requested series.
        mapping_data = json.load(series_mapping)
        variableId_list = [mapping['variableId'] for mapping in mapping_data]
        if str(variableId) not in variableId_list:
            raise Exception("Requested series not mapped into the Fingrid mapping file!")
        else:
            mapping = [mapping for mapping in mapping_data if mapping['variableId'] == variableId][0]
    if mapping['time_frequency'].split()[1] == 'min':  # If time frequency in minutes, split the request to parts.
        series_data = pd.DataFrame()
        for start_end_tuple in star_end_dates.get(mapping['time_frequency'].split()[0]):
            start_time, end_time = start_end_tuple[0], start_end_tuple[1]
            series_data_request = request_data(start_time=start_time, end_time=end_time,
                                               year=year, variableId=variableId)
            if series_data_request.empty:
                print(variableId + ' unavailable from {} to {}'.format(start_time, end_time))
                continue
            series_data = series_data.append(series_data_request)
            time.sleep(1)
    else:  # Else request whole year
        start_time = r'-01-01T00:00:00Z'
        end_time = r'-12-31T00:00:00Z'
        series_data = request_data(start_time=start_time, end_time=end_time,
                                   year=year, variableId=variableId)
    return series_data


def get_full_series(variableId):
    """ Request years in range for a single series """

    series_data = pd.DataFrame()
    for year in range(update_year, update_year + 1):
        yearly_data = get_full_year(variableId=variableId, year=year)
        if yearly_data.empty:
            print(variableId + ' unavailable for ' + str(year))
            continue
        series_data = series_data.append(yearly_data)
        time.sleep(5)
    series_data['variableId'] = variableId
    return series_data


def insert_full_series_to_db(variableId):
    """ Insert full series with variableId """

    data = get_full_series(variableId=variableId)

    sqlite_file = r'data\fingrid_open_data.db'
    conn = sqlite3.connect(sqlite_file)
    sql = """SELECT * FROM SeriesFact WHERE variableId = {} and substr(Datetime, 1, 4) == '{}'""".format(variableId,
                                                                                                         update_year)
    existing_data = pd.read_sql(sql, con=conn)
    existing_data.index = existing_data['Datetime']
    data = data[~data.index.isin(existing_data.index)]

    data.to_sql(name="SeriesFact", con=conn, if_exists="append")

def insert_all_to_db():
    """ Fill the whole sqlite database """

    with open(r'data_mappings/fingrid_fill.json') as series_mapping:
        mapping_data = json.load(series_mapping)
        variableId_list = [mapping['variableId'] for mapping in mapping_data]
    for variableId in variableId_list:
        print('Inserting ' + str(variableId))
        insert_full_series_to_db(variableId)
