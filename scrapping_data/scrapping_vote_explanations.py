import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By

import time

import pandas as pd
import json
import numpy as np

from tqdm import tqdm

# driver = webdriver.Chrome()
# driver.get(page_to_scrap)
# time.sleep(2)

# #Accept cookies 
# driver.find_elements(By.CSS_SELECTOR, 'button.epjs_agree:nth-child(2)')[0].click()
# time.sleep(2)



list_to_iter_on = pd.read_csv('../data/liste_meps.csv')
list_to_iter_on=list_to_iter_on[list_to_iter_on.country=="France"]
list_to_iter_on.rename(columns={'Unnamed: 0':"name"}, inplace=True)
list_to_iter_on.name = list_to_iter_on.name.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

driver = webdriver.Firefox()

data={}
i = 0
for index, row in tqdm(list_to_iter_on.iterrows()) : 
    i +=1
    page_to_scrap = f"https://www.europarl.europa.eu/meps/fr/{row['number']}/{row['name']}/other-activities/written-explanations#detailedcardmep"
    driver.get(page_to_scrap)
    time.sleep(np.random.uniform(2,3))

    #Accept cookies 
    if len(driver.find_elements(By.CSS_SELECTOR, 'button.epjs_agree:nth-child(2)')) > 0:
        driver.find_elements(By.CSS_SELECTOR, 'button.epjs_agree:nth-child(2)')[0].click()
        time.sleep(2)

    #if url equals page_to_scrap, we are on the right page, if not get to next row
    if driver.current_url != page_to_scrap:

        data[row["name"]]  = {"source" : [], "contenu": ["Pas d'explications écrites"], "source_date": []}
        print("No data for ", row["name"])
        time.sleep(np.random.uniform(2,3))
        continue


    while len(driver.find_elements(By.CSS_SELECTOR, '.btn-default')) > 0:
        driver.find_elements(By.CSS_SELECTOR, '.btn-default')[0].click()
        time.sleep(np.random.uniform(1,1.5))

    #Get the data
    titres = driver.find_elements(By.CLASS_NAME, 'erpl_document-header')
    contenu = driver.find_elements(By.CLASS_NAME, 'erpl_document-body > p')
    dates = driver.find_elements(By.CLASS_NAME, 'erpl_document-subtitle-fragment')

    data[row["name"]] = {"source" : [titres[i].text for i in range(len(titres))], "contenu": [contenu[i].text for i in range(len(titres))], "source_date": [dates[i].text for i in range(len(titres))]}
    time.sleep(np.random.uniform(2,3))

    # all 10 index , save the data
    if i % 3 == 0:
        pd.DataFrame.from_dict(data).T.explode(column=["source", "contenu", "source_date"]).to_csv(f"../data/all_french_explanations_{index}.csv")
        data={}
    #pd.DataFrame.from_dict(data).T.explode(column=["source", "contenu", "source_date"]).to_csv("../data/all_french_explanations.csv")

#merge all files that begins with all_french_explanations 




# #choice of MEP to scrap
# num_MEP = "135511"
# name_MEP = "VALERIE_HAYER"
# page_to_scrap = f"https://www.europarl.europa.eu/meps/fr/{num_MEP}/{name_MEP}/other-activities/written-explanations#detailedcardmep"

# driver = webdriver.Chrome()
# driver.get(page_to_scrap)
# time.sleep(2)

# #Accept cookies 
# driver.find_elements(By.CSS_SELECTOR, 'button.epjs_agree:nth-child(2)')[0].click()
# time.sleep(2)

# #Expand all the written explanations
# while len(driver.find_elements(By.CSS_SELECTOR, '.btn-default')) > 0:
#     driver.find_elements(By.CSS_SELECTOR, '.btn-default')[0].click()
#     time.sleep(1)

# #Get the data
# titres = driver.find_elements(By.CLASS_NAME, 'erpl_document-header')
# contenu = driver.find_elements(By.CLASS_NAME, 'erpl_document-body > p')
# dates = driver.find_elements(By.CLASS_NAME, 'erpl_document-subtitle-fragment')

# data = {name_MEP: [{"source" : titres[i].text, "contenu": contenu[i].text, "source_date": dates[i].text} for i in range(len(titres))]}

# #save the data
# pd.DataFrame(data[name_MEP]).to_csv(f"../data/{name_MEP}.csv", index=False)
