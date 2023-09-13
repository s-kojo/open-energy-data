import pandas as pd
import os
import json


def convert_weekly_to_hourly(weekly_data):
    weekly_data['Week'] = [week_num[0] for week_num in weekly_data['Unnamed: 0_level_2'].str.split()]
    weekly_data['Year'] = ['20' + week_num[2] for week_num in weekly_data['Unnamed: 0_level_2'].str.split()]
    weekly_data.drop(columns=[r'Unnamed: 0_level_2'], inplace=True)
    weekly_data['date'] = pd.to_datetime(weekly_data.Year.astype(str), format='%Y') + \
                          pd.to_timedelta(weekly_data.Week.astype(int).mul(7).astype(str) + ' days')
    hours = pd.date_range(min(weekly_data['date']), max(weekly_data['date']), freq='H')
    weekly_data = weekly_data.set_index('date')
    weekly_data = weekly_data.reindex(hours)
    weekly_data = weekly_data.interpolate(method='linear', limit_direction='forward', axis=0)
    weekly_data.drop(columns=['Week', 'Year'], inplace=True)
    weekly_data.index.rename('DateTime', inplace=True)
    return weekly_data


def convert_hourly_to_datetime(hourly_data):
    hourly_data['Hours'] = [hours[0] for hours in hourly_data.Hours.str.split()]
    hourly_data['DateTime'] = pd.to_datetime(hourly_data['Unnamed: 0_level_2'] + "-" + hourly_data['Hours'].astype(str),
                                             format='%d-%m-%Y-%H')
    hourly_data = hourly_data.set_index('DateTime')
    hourly_data = hourly_data.drop(columns=['Unnamed: 0_level_2', 'Hours'])
    return hourly_data


def get_year(file_name):
    try:
        series_data = pd.read_html(file_name, decimal=',', thousands=' ')[0]
        if max([len(column) for column in series_data.columns]) == 3:
            series_data.columns = [column[2] for column in series_data.columns]
        else:
            series_data.columns = [column[2] + ', ' + column[3] for column in series_data.columns]
            series_data.rename(columns={"Unnamed: 0_level_2, Unnamed: 0_level_3": "Unnamed: 0_level_2",
                                        "Unnamed: 1_level_2, Hours": "Hours"}, inplace=True)
    except IndexError:
        series_data = pd.DataFrame()
    return series_data

def get_full_series(series):
    with open(r'data_mappings\nordpool_dev.json') as series_mapping:
        mapping_data = json.load(series_mapping)
        mapping = [mapping for mapping in mapping_data if mapping['name'] == series][0]
    name = mapping["name"]
    series_data = pd.DataFrame()

    for year in range(2013, 2022):
        try:
            file_name = os.path.join('nordpool_open_data', mapping['folder'],
                                         mapping["file_and_sheet_name"].format(str(year)) + '.xls')
            print(file_name)
            yearly_data = get_year(file_name)
            yearly_data.columns = yearly_data.columns.map(lambda x: x.replace('.1', ''))
            yearly_data = yearly_data.groupby(yearly_data.columns, axis=1).sum()
            if yearly_data.empty:
                continue
            if mapping['data_period'] == '1 h':
                yearly_data = convert_hourly_to_datetime(yearly_data)
            elif mapping['data_period'] == '1 w':
                yearly_data = convert_weekly_to_hourly(yearly_data)
            else:
                raise Exception('No conversion defined for Date period {}'.format(mapping['data_period']))
            yearly_data.index = yearly_data.index.tz_localize('Europe/Brussels',
                                                              ambiguous='NaT', nonexistent='NaT').\
                                                              tz_convert('UTC')
            yearly_data = yearly_data.loc[yearly_data.index.notnull()]
            yearly_data.columns = [name + ', ' + column for column in yearly_data.columns]
            series_data = series_data.append(yearly_data, sort=True)
        except ValueError:
            print("error with series {}".format(file_name))
            continue

    return series_data


def get_all_data():
    with open(r'data_mappings\nordpool_dev.json') as series_mapping:
        mapping_data = json.load(series_mapping)
        series = [mapping['name'] for mapping in mapping_data]
    full_data = get_full_series(series[0])
    i = 1
    for variable in series[1:]:
        print(str(i) + " out of " + str(len(series)) + " downloaded")
        i += 1
        series_data = get_full_series(variable)
        full_data = full_data.join(series_data, how='outer', rsuffix='_double')
        del series_data
    full_data = full_data[full_data.columns[~full_data.columns.isin(full_data.filter(like='_double').columns)]]
    return full_data

def write_all_data():
    full_data = get_all_data()
    full_data.to_csv("data/nordpool_data.csv", index=True, sep='|')
