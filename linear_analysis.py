import pandas as pd
import numpy as np
import sqlite3
import holidays
from functools import reduce
import statsmodels.api as sm
from sklearn.metrics import r2_score
import re


''' Fingrid data '''


def filter_outliers(series):
    series[series > series.quantile(0.99) + 2 * series.std()] = np.NaN
    return series


def correct_timezone(series):
    series_eet = series[series.index < pd.to_datetime('2017-02-16T00:00:00Z')]
    series_utc = series[series.index >= pd.to_datetime('2017-02-16T00:00:00Z')]
    series_eet = series_eet.tz_localize(None).tz_localize('EET', ambiguous='NaT').tz_convert('UTC')
    return pd.concat([series_eet, series_utc])


def get_series(series_id, filter_out=False):
    series = pd.read_sql(sql=select_data_frame.format(series_id), con=conn,
                         index_col='Datetime', parse_dates=['Datetime'])
    series.rename(columns={'value': series_id}, inplace=True)
    series = series[~series.index.duplicated()]
    if filter_out:
        series = filter_outliers(series)
    series = series.resample('1h').mean()
    return series


fingrid_variables = ['243', '241', '242', '166', '165', '124', '245', '64', "247", '267', '74',
                     '246', '181', '30', '29', '28', '24', '25', '31', '34', '32', '111',
                     '191', '188', '205', '202', '266', '201', '268', '189', '248',
                     '55', '58', '60', '57', '61']


time_zone_corrected = ['191', '188', '181', '248', '205', '201', '189', '202']

sqlite_file = "./data/fingrid_open_data.db"
conn = sqlite3.connect(sqlite_file)
select_data_frame = "Select Datetime, value From SeriesFact Where variableId == {}"

fingrid_data = [get_series(series_id) for series_id in fingrid_variables]
fingrid_data = reduce(lambda df1, df2: pd.merge(df1, df2, on='Datetime', how='outer'), fingrid_data)

''' Nordpool '''

nordpool_data = pd.read_csv("./data/nordpool_data_filtered.csv", sep='|', index_col=[0])
nordpool_data.index = pd.to_datetime(nordpool_data.index)
non_zero_columns = nordpool_data.filter(regex='^((?!hinta).)*$').columns
nordpool_data[non_zero_columns] = nordpool_data[non_zero_columns].replace({'0': np.nan, 0: np.nan})

y = np.log2(np.log2(nordpool_data['Saatohinta, EUR/MWh, FI, Up'] - nordpool_data['Elspot-hinta, EUR/MWh, FI'] + 1) + 1).dropna()
y.name = 'log-hinta'

X = pd.DataFrame(index=y.index)

X['vuosi'] = X.index.year - X.index.year.min()  # Käyttäytyy melko lineaarisesti

fi_holidays = holidays.FI()
X['loma'] = np.minimum(1, [date in fi_holidays for date in X.index.date] + np.isin(X.index.dayofweek, [5, 6]).astype(int))

X['Vesivoimareservi, FI'] = nordpool_data['Vesivoimareservi, GWh, FI']
X['Vesivoimareservi, NO'] = nordpool_data['Vesivoimareservi, GWh, NO']
X['Vesivoimareservi, SE'] = nordpool_data['Vesivoimareservi, GWh, SE']

''' Country differences'''

difference_variables = {'Tuulituotantoennuste': 'Tuulituotanto,', 'Kulutusennuste': 'Kulutus,',
                        'Tuotantoennuste': 'Tuotanto,', 'Kulutus,': 'Tuotanto,'}

for key, value in difference_variables.items():
    forecast_series = nordpool_data.filter(like=key)
    forecast_series = forecast_series[forecast_series.columns.drop(list(forecast_series.filter(like='FI')))]
    forecast_series = forecast_series.reindex(sorted(forecast_series.columns), axis=1)
    value_series = nordpool_data.filter(like=value)
    value_series = value_series.reindex(sorted(value_series.columns), axis=1)
    value_series = value_series[value_series.columns.drop(list(value_series.filter(like='FI')))]
    if key == 'Kulutusennuste':
        forecast_series.columns = forecast_series.columns.str.split(',').map(lambda x: re.sub(r"\d+", " ", ''.join(x[:3])))
        value_series.columns = value_series.columns.str.split(',').map(lambda x: re.sub(r"\d+", " ", ''.join(x[:3])))
        forecast_series = forecast_series.T.groupby(lambda x: x).sum().T
        value_series = value_series.T.groupby(lambda x: x).sum().T
    common_index = forecast_series.index.union(value_series.index)
    forecast_series = forecast_series.loc[common_index]
    forecast_series.replace(0, np.nan, inplace=True)
    value_series = value_series.loc[common_index]
    value_series.replace(0, np.nan, inplace=True)
    difference_df = pd.DataFrame(data=forecast_series.values - value_series.values,
                                 index=common_index,
                                 columns=list(map(lambda x: key + ' virhe ' + x.split(',')[-1], value_series.columns.tolist())))
    X = X.join(difference_df, how='left')

''' Fingrid '''

X['Poikkeama tuotantosuunnitelmasta'] = fingrid_data['242'] - fingrid_data["74"]
X['Tuotantoennuste - lyhyt - pitka'] = fingrid_data['241'] - fingrid_data['242']

X['Kulutusennuste - virhe'] = fingrid_data['166'] - fingrid_data['124']
X['Kulutusennuste - lyhyt - pitka'] = fingrid_data['166'] - fingrid_data['165']
X['Kokonaiskulutus'] = fingrid_data['124']

X['Tuuliennuste - virhe'] = fingrid_data['245'] - fingrid_data['181']
X['Tuuliennuste - lyhyt - pitka'] = fingrid_data['245'] - fingrid_data['246']
X['Aurinkoennuste - lyhyt - pitka'] = fingrid_data['247'] - fingrid_data['248']

X['Vapaa P1 kapasiteetti etelasta pohjoiseen'] = fingrid_data['30'] - fingrid_data['29']
X['Vapaa P1 kapasiteetti pohjoisesta etelaan'] = fingrid_data['28'] - fingrid_data['30']
X['SE1 vapaakapasiteetti'] = fingrid_data['24'] + fingrid_data['31']
X['SE3 vapaakapasiteetti'] = fingrid_data['25'] + fingrid_data['32']
X['RU vapaakapasiteetti'] = fingrid_data['64']
X['EE vapaakapasiteetti'] = fingrid_data['111']

X["Sahkonsiirto FI-SE3"] = fingrid_data["61"]
X["Sahkonsiirto FI-NO"] = fingrid_data["57"]
X["Sahkonsiirto FI-SE1"] = fingrid_data["60"]
X["Sahkonsiirto FI-RUS"] = fingrid_data["58"]
X["Sahkonsiirto FI-EE"] = fingrid_data["55"]

X['Vesivoimatuotanto (kapasiteetista)'] = fingrid_data['191'] / fingrid_data['191'].rolling(window=pd.Timedelta('3652D')).max()
X['Ydinvoimatuotannon (kapasiteetista)'] = fingrid_data['188'] / fingrid_data['188'].rolling(window=pd.Timedelta('3652D')).max()
X['Tuulituotannon osuus (kapasiteetista)'] = fingrid_data['181'] / fingrid_data['268']
X['Aurinkotuotannon osuus (kapasiteetista)'] = fingrid_data['248'] / fingrid_data['267']
X['Pientuotannon osuus (kapasiteetista)'] = fingrid_data['205'] / fingrid_data['205'].rolling(window=pd.Timedelta('365D')).max()
X['CHP:n osuus'] = (fingrid_data['201'] + fingrid_data['189'].fillna(0)) / (fingrid_data['201'] + fingrid_data['189'].fillna(0)).rolling(window=pd.Timedelta('365D')).max()
X['Teollisuudentuotannon osuus'] = fingrid_data['202'] / fingrid_data['202'].rolling(window=pd.Timedelta('365D')).max()
X['Tuotannon päästökerroin'] = fingrid_data['266']
X["Ylossaatotarjousten summa"] = fingrid_data['243']
X['Elspot-hinta'] = nordpool_data['Elspot-hinta, EUR/MWh, FI']

UMM_data = pd.read_csv(r"./data/UMM_data_aggregate.csv", sep='|', index_col=[0])
UMM_data.index = pd.to_datetime(UMM_data.index)

X['Tuotanto UMM FI (ei suunniteltu)'] = UMM_data.filter(regex='FI [A-Za-z0-9]* unplannedproduction').sum(axis=1)
X['Tuotanto UMM FI (suunniteltu)'] = UMM_data.filter(regex='FI [A-Za-z0-9]* plannedproduction').sum(axis=1)
X['Ydinvoima UMM FI (ei suunniteltu)'] = UMM_data['UMMs FI Nuclear unplannedproduction']
X['Ydinvoima UMM FI (suunniteltu)'] = UMM_data['UMMs FI Nuclear plannedproduction']

X['Siirto UMM FI (ei suunniteltu)'] = UMM_data.filter(like='- FI unplanned').sum(axis=1)
X['Siirto UMM FI (suunniteltu)'] = UMM_data.filter(like='- FI planned').sum(axis=1)

X['Tuotanto summa NP (ei suunniteltu)'] = UMM_data.filter(like='unplannedproduction').sum(axis=1)
X['Tuotanto summa NP (suunniteltu)'] = UMM_data.filter(like='plannedproduction').sum(axis=1)
X['Ydinvoima UMM NP (ei suunniteltu)'] = UMM_data.filter(like='Nuclear unplannedproduction').sum(axis=1)
X['Ydinvoima UMM FI (suunniteltu)'] = UMM_data.filter(like='Nuclear plannedproduction').sum(axis=1)


X[X == np.Inf] = np.nan

common_index = y.dropna(how='any').index.intersection(X.dropna(how='any').index)

y = y.loc[common_index]
X = X.loc[common_index]

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

pred = pd.DataFrame(data={'ennuste': est2.predict(X2)}, index=y.index)
pred[pred['ennuste'] < 0] = 0

pred = pred.join(y)

print('Linear R2 {}'.format(r2_score(y, pred['ennuste'])))


