import pandas as pd

UMM_data = pd.read_csv(r"C:\Users\ksamko\Python Projects\Fingrid Modeling\Data\UMM_data.csv", sep='|')

UMM_data["Unavailable Capacity"] = UMM_data["Unavailable Capacity"].str.replace(' MW', '').astype(float)
UMM_data["Unavailable Capacity"] = UMM_data["Unavailable Capacity"].abs()
UMM_data['From'] = pd.to_datetime(UMM_data['From'], format = '%d.%m.%Y %H:%M').dt.tz_localize('Europe/Oslo', ambiguous = True).dt.tz_convert('UTC')
UMM_data['To'] = pd.to_datetime(UMM_data['To'], format = '%d.%m.%Y %H:%M').dt.tz_localize('Europe/Oslo', ambiguous = True).dt.tz_convert('UTC')
t1 = UMM_data['From'].min()
t2 = UMM_data['To'].max()
idx = pd.date_range(t1.ceil('1h'), t2.ceil('1h'), freq='1h')

UMM_df = pd.DataFrame(index=idx)

for unavailibility_type in UMM_data['unavailibility type'].unique():
    for message_type in UMM_data['message type'].unique():
        for area in UMM_data['Area'].unique():
            for fuel in UMM_data['Fuel Type'].unique():
                if str(area) == 'nan' or str(message_type) == 'nan' or str(unavailibility_type) == 'nan' or str(fuel) == 'nan':
                    continue
                filtered_UMM_data = UMM_data[(UMM_data['unavailibility type'] == unavailibility_type) &
                                             (UMM_data['message type'] == message_type) & (UMM_data['Area'] == area) &
                                             (UMM_data['Fuel Type'] == fuel)]
                i = 0
                UMM_df_temp = pd.DataFrame(index=idx)
                UMM_df["UMMs " + area + " " + fuel + " " + unavailibility_type + message_type] = 0
                for uc, From, To in zip(filtered_UMM_data["Unavailable Capacity"], filtered_UMM_data['From'], filtered_UMM_data['To']):
                    UMM_df_temp = pd.DataFrame(index=idx, columns=['Sum_column'])
                    UMM_df_temp['Sum_column'] = 0
                    UMM_df_temp['Sum_column'][(idx > From) & (idx <= To)] = uc
                    UMM_df["UMMs " + area + " " + fuel + " " + unavailibility_type + message_type] = \
                        UMM_df["UMMs " + area + " " + fuel + " " + unavailibility_type + message_type] + UMM_df_temp['Sum_column']
                    i += 1

UMM_transmission_data = pd.read_csv(r"C:\Users\ksamko\Python Projects\Fingrid Modeling\Data\UMM_transmission_data.csv", sep='|')

UMM_transmission_data["Unavailable Capacity"] = UMM_transmission_data["Unavailable Capacity"].str.replace(' MW', '').astype(float)
UMM_transmission_data["Unavailable Capacity"] = UMM_transmission_data["Unavailable Capacity"].abs()
UMM_transmission_data['From'] = pd.to_datetime(UMM_transmission_data['From'], format = '%d.%m.%Y %H:%M').dt.tz_localize('Europe/Oslo', ambiguous = True).dt.tz_convert('UTC')
UMM_transmission_data['To'] = pd.to_datetime(UMM_transmission_data['To'], format = '%d.%m.%Y %H:%M').dt.tz_localize('Europe/Oslo', ambiguous = True).dt.tz_convert('UTC')
t1 = UMM_transmission_data['From'].min()
t2 = UMM_transmission_data['To'].max()
idx = pd.date_range(t1.ceil('1h'), t2.ceil('1h'), freq='1h')

UMM_transmission_df = pd.DataFrame(index=idx)

for unavailibility_type in UMM_transmission_data['unavailibility type'].unique():
    for area in UMM_transmission_data['Area'].unique():
        if str(area) == 'nan' or str(unavailibility_type) == 'nan':
            continue
        filtered_UMM_data = UMM_transmission_data[(UMM_transmission_data['unavailibility type'] == unavailibility_type) &
                                                  (UMM_transmission_data['Area'] == area)]
        i = 0
        UMM_df_temp = pd.DataFrame(index=idx)
        UMM_transmission_df["UMMs " + area + " " + unavailibility_type] = 0
        for uc, From, To in zip(filtered_UMM_data["Unavailable Capacity"], filtered_UMM_data['From'],
                                filtered_UMM_data['To']):
            UMM_df_temp = pd.DataFrame(index=idx, columns=['Sum_column'])
            UMM_df_temp['Sum_column'] = 0
            UMM_df_temp['Sum_column'][(idx > From) & (idx <= To)] = uc
            UMM_transmission_df["UMMs " + area + " " + unavailibility_type] = \
                UMM_transmission_df["UMMs " + area + " " + unavailibility_type] + UMM_df_temp['Sum_column']
            i += 1

UMM_df = UMM_df.join(UMM_transmission_df, how='outer')

UMM_df.to_csv(r"data\UMM_data_aggregate.csv", sep='|')