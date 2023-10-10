import pandas as pd
import numpy as np
import sqlite3
import holidays
from functools import reduce
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import shap
import re
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import gridspec

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
nordpool_data[non_zero_columns] = nordpool_data[non_zero_columns].replace({'0':np.nan, 0:np.nan})

# y = np.log2(np.log2(nordpool_data['Saatohinta, EUR/MWh, FI, Up'] - nordpool_data['Elspot-hinta, EUR/MWh, FI'] + 1) + 1).dropna()
# y.name = 'log-log-hinta'
y = (nordpool_data['Saatohinta, EUR/MWh, FI, Up'] - nordpool_data['Elspot-hinta, EUR/MWh, FI']).dropna()
y.name = 'hinta'

X = pd.DataFrame(index=y.index)
X['vuosi'] = X.index.year - X.index.year.min()

fi_holidays = holidays.FI()
X['loma'] = np.minimum(1, [date in fi_holidays for date in X.index.date] + np.isin(X.index.dayofweek, [5, 6]).astype(int))

X['Vuosi sykli 1'] = np.sin(2 * np.pi * (24 * X.index.dayofyear + X.index.hour) / 8670)
X['Vuosi sykli 2'] = np.cos(2 * np.pi * (24 * X.index.dayofyear + X.index.hour) / 8670)

X['Tunti sykli 1'] = np.sin(2 * np.pi * (X.index.hour) / 24)
X['Tunti sykli 2'] = np.cos(2 * np.pi * (X.index.hour) / 24)

X['Vesivoimareservi, FI'] = nordpool_data['Vesivoimareservi, GWh, FI']
X['Vesivoimareservi, NO'] = nordpool_data['Vesivoimareservi, GWh, NO']
X['Vesivoimareservi, SE'] = nordpool_data['Vesivoimareservi, GWh, SE']
X['Aluehintaero'] = nordpool_data['Elspot-hinta, EUR/MWh, SYS'] - nordpool_data['Elspot-hinta, EUR/MWh, FI']

''' Country differences'''

difference_variables = {'Tuulituotantoennuste': 'Tuulituotanto,', 'Kulutusennuste': 'Kulutus,'}
                        # ,'Tuotantoennuste': 'Tuotanto,', 'Kulutus,': 'Tuotanto,'}

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

# X['Poikkeama tuotantosuunnitelmasta'] = fingrid_data['242'] - fingrid_data["74"]
# X['Tuotantoennuste - lyhyt - pitka'] = fingrid_data['241'] - fingrid_data['242']

X['Kulutusennuste - virhe'] = fingrid_data['166'] - fingrid_data['124']
X['Kulutusennuste - lyhyt - pitka'] = fingrid_data['166'] - fingrid_data['165']
X['Kokonaiskulutus'] = fingrid_data['124']

X['Tuuliennuste - virhe'] = fingrid_data['245'] - fingrid_data['181']
X['Tuuliennuste - lyhyt - pitka'] = fingrid_data['245'] - fingrid_data['246']
X['Aurinkoennuste - lyhyt - pitka'] = fingrid_data['247'] - fingrid_data['248']

X['Vapaa P1 kapasiteetti etelasta pohjoiseen'] = fingrid_data['30'] - fingrid_data['29']
X['Vapaa P1 kapasiteetti pohjoisesta etelaan'] = fingrid_data['28'] - fingrid_data['30']

fingrid_data['24'][-fingrid_data['31'] > fingrid_data['24']] = np.nan
fingrid_data['25'][-fingrid_data['32'] > fingrid_data['25']] = np.nan

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

X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
y_train, y_test = y[y.index.year != 2022], y[y.index.year == 2022]
X_train, X_test = X[X.index.year != 2022], X[X.index.year == 2022]

model = LGBMRegressor(n_estimators=6000)
model.fit(X_train, y_train)

pred = pd.DataFrame(data={'ennuste': model.predict(X_test)}, index=y_test.index)
pred[pred['ennuste'] < 0] = 0

pred = pred.join(y_test)

print('LGB R2 {}'.format(r2_score(pred['hinta'], pred['ennuste'])))

X_sampled = X_test

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sampled)
shap_interaction = explainer.shap_interaction_values(X_sampled)

# Get absolute mean of matrices
mean_shap = np.quantile(np.abs(shap_interaction), q=0.99, axis=0)
df = pd.DataFrame(mean_shap, index=X.columns, columns=X.columns)

# times off diagonal by 2
df.where(df.values == np.diagonal(df), df.values*2, inplace=True)

# display
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot()
sns.heatmap(df.round(decimals=3), cmap='magma', annot=False, fmt='.6g', cbar=True, ax=ax, )
ax.tick_params(axis='x', labelsize=10, rotation=90)
ax.tick_params(axis='y', labelsize=10)

plt.yticks(rotation=0)
plt.show()

# plot feature interaction
def plot_feature_interaction(f1, f2, percentile=10):
    # dependence plot
    fig = plt.figure(tight_layout=True, figsize=(20, 10))
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    ax0 = fig.add_subplot(spec[0, 0])
    lowv, highv = np.percentile(X[f2].dropna(), [percentile, 100-percentile])

    shap.dependence_plot(f1, shap_values.values, X_sampled, display_features=X_sampled, interaction_index=f2, ax=ax0,
                         show=False)
    ax0.set_title(f'SHAP main effect', fontsize=10)

    ax1 = fig.add_subplot(spec[0, 1])
    shap.dependence_plot((f1, f2), shap_interaction, X_sampled, display_features=X_sampled, ax=ax1, axis_color='w',
                         show=False)
    ax1.set_title(f'SHAP interaction effect', fontsize=10)

    temp = pd.DataFrame({f1: X[f1].values,
                         'target': y.values})
    temp = temp.sort_values(f1)
    temp.reset_index(inplace=True)

    ax3 = fig.add_subplot(spec[1, 0])
    sns.scatterplot(x=temp[f1], y=temp.target.rolling(3000, center=True).mean(), data=temp, ax=ax3, s=2)
    ax3.set_title(f'Target (rolling mean) on {f1}', fontsize=10)

    temp = pd.DataFrame({f1: X.loc[X[f2] < lowv, f1].values,
                         'target': y.loc[X[f2] < lowv].values})
    temp = temp.sort_values(f1)
    temp.reset_index(inplace=True)

    ax4 = fig.add_subplot(spec[1, 1])
    sns.scatterplot(x=temp[f1], y=temp.target.rolling(3000, center=True).mean(), data=temp, ax=ax4, s=2)
    ax4.set_title(f'Target (rolling mean) on {f1} & {f2} below {str(percentile)}th percentile',
                  fontsize=10)

    temp = pd.DataFrame({f1: X.loc[X[f2] > highv, f1].values,
                         'target': y.loc[X[f2] > highv].values})

    temp = temp.sort_values(f1)
    temp.reset_index(inplace=True)

    ax5 = fig.add_subplot(spec[1, 2])
    sns.scatterplot(x=temp[f1], y=temp.target.rolling(3000, center=True).mean(), data=temp, ax=ax5, s=2)
    ax5.set_title(f'Target (rolling mean) on {f1} & {f2} above {str(100 - percentile)}th percentile',
                  fontsize=10)

    plt.suptitle(f"Feature Interaction Analysis\n {f1} & {f2}", fontsize=30, y=1.15)
    fig.tight_layout()

    plt.show()

f1 = 'VesivoimareserviSE'
f2 = 'VesivoimareserviNO'
percentile = 10

plot_feature_interaction(f1, f2, percentile)
