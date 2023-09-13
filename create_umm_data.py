import pandas as pd
import time
import requests
from bs4 import BeautifulSoup

base_url = "http://ummrss.nordpoolgroup.com/messages/?areas=10YDK-1--------W&areas=10YDK-2--------M&areas=10YFI-1--------U&areas=10YNO-1--------2&areas=10YNO-2--------T&areas=10YNO-3--------J&areas=10YNO-4--------9&areas=10Y1001A1001A48H&areas=10Y1001A1001A44P&areas=10Y1001A1001A45N&areas=10Y1001A1001A46L&areas=10Y1001A1001A47J&limit=10000&messageTypes={}&eventStartDate={}-01-01T00:00:00.000Z&eventStopDate={}-12-31T23:00:00.000Z&status=1&unavailabilityType={}"

years = [year for year in range(2013, 2024)]
unavailibilityTypes = {"1": "unplanned", "2": "planned"}
messageTypes = {"1": "production", "2": "consumption"}

scraped_data = []
for messageType in messageTypes:
    for unavailibilityType in unavailibilityTypes:
        for year in years:
            r = requests.get(base_url.format(messageType, year, year, unavailibilityType))
            soup = BeautifulSoup(r.content.decode('utf-8-sig','ignore'))
            soup = soup.prettify(formatter=None)
            message_list = pd.read_html(soup)
            message_list = [message_list[i].ffill().bfill().iloc[0] for i in range(len(message_list))
                            if message_list[i].shape[1] != 2]
            message_df = pd.concat(message_list, axis=1).T
            message_df["unavailibility type"] = unavailibilityTypes[unavailibilityType]
            print(message_df.__len__())
            message_df["message type"] = messageTypes[messageType]
            scraped_data.append(message_df)
            time.sleep(1)

scraped_data = pd.concat(scraped_data)
scraped_data.to_csv(r"data/UMM_data.csv", sep = '|', index=False)