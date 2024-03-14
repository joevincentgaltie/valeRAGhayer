import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By

import time

import pandas as pd
import json



#choice of MEP to scrap
num_MEP = "135511"
name_MEP = "VALERIE_HAYER"
page_to_scrap = f"https://www.europarl.europa.eu/meps/fr/{num_MEP}/{name_MEP}/other-activities/written-explanations#detailedcardmep"



driver = webdriver.Firefox()
driver.get(page_to_scrap)
time.sleep(2)

#Accept cookies 
driver.find_elements(By.CSS_SELECTOR, 'button.epjs_agree:nth-child(2)')[0].click()
time.sleep(2)

#Expand all the written explanations
while len(driver.find_elements(By.CSS_SELECTOR, '.btn-default')) > 0:
    driver.find_elements(By.CSS_SELECTOR, '.btn-default')[0].click()
    time.sleep(1)

#Get the data
titres = driver.find_elements(By.CLASS_NAME, 'erpl_document-header')
contenu = driver.find_elements(By.CLASS_NAME, 'erpl_document-body > p')
dates = driver.find_elements(By.CLASS_NAME, 'erpl_document-subtitle-fragment')

data = {name_MEP: [{"source" : titres[i].text, "contenu": contenu[i].text, "source_date": dates[i].text} for i in range(len(titres))]}

#save the data
pd.DataFrame(data[name_MEP]).to_csv(f"../data/{name_MEP}.csv", index=False)
