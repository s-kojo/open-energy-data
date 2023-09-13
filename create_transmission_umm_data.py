import pandas as pd
import time
import requests
from bs4 import BeautifulSoup

base_url = "https://ummrss.nordpoolgroup.com/messages/?connections=EE-FI&connections=FI-EE&connections=FI-RU&connections=FI-SE1&connections=FI-SE3&connections=RU-FI&connections=SE1-FI&connections=SE3-FI&eventStartDate={}-01-01T00:00:00.000Z&eventStopDate={}-12-31T23:00:00.000Z&limit=1000&status=1&unavailabilityType={}"

years = [year for year in range(2013, 2024)]
unavailibilityTypes = {"1": "unplanned", "2": "planned"}

scraped_data = []
for unavailibilityType in unavailibilityTypes:
    for year in years:
        r = requests.get(base_url.format(year, year, unavailibilityType))
        soup = BeautifulSoup(r.content.decode('utf-8-sig', 'ignore'))
        soup = soup.prettify(formatter=None)
        message_list = pd.read_html(soup)
        message_list = [message_list[i].ffill().bfill().iloc[0] for i in range(len(message_list))
                        if message_list[i].shape[1] != 2]
        message_df = pd.concat(message_list, axis=1).T
        message_df["unavailibility type"] = unavailibilityTypes[unavailibilityType]
        scraped_data.append(message_df)
        time.sleep(1)

scraped_data = pd.concat(scraped_data)
scraped_data.to_csv(r"data/UMM_transmission_data.csv", sep='|', index=False)