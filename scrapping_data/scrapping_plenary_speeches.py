import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By

import time

import pandas as pd
import json
import numpy as np
import os

from tqdm import tqdm


list_to_iter_on = pd.read_csv('../data/liste_meps.csv')
list_to_iter_on=list_to_iter_on[list_to_iter_on.country=="France"]
list_to_iter_on.rename(columns={'Unnamed: 0':"name"}, inplace=True)
list_to_iter_on.name = list_to_iter_on.name.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')


driver = webdriver.Chrome()


results = pd.DataFrame(columns=["name", "source", "contenu", "source_date"])
i = 0
for index, row in tqdm(list_to_iter_on.iterrows()) : 
    i +=1
    page_to_scrap = f"https://www.europarl.europa.eu/meps/fr/{row['number']}/{row['name']}/main-activities/plenary-speeches#detailedcardmep"
    driver.get(page_to_scrap)
    time.sleep(np.random.uniform(4,5))

     #Accept cookies 
    if len(driver.find_elements(By.CSS_SELECTOR, 'button.epjs_agree:nth-child(2)')) > 0:
        driver.find_elements(By.CSS_SELECTOR, 'button.epjs_agree:nth-child(2)')[0].click()
        time.sleep(2)

    if driver.current_url != page_to_scrap:

        #data[row["name"]]  = {"source" : [], "contenu": ["Pas d'explications écrites"], "source_date": []}
        print("No data for ", row["name"])
        time.sleep(np.random.uniform(2,3))
        continue

    while len(driver.find_elements(By.CSS_SELECTOR, '.btn-default')) > 0:
        driver.find_elements(By.CSS_SELECTOR, '.btn-default')[0].click()
        time.sleep(np.random.uniform(1,1.5))


    titres = [ele.text for ele in driver.find_elements(By.CLASS_NAME, 'erpl_document-header > h3 > a > span')]
    

    
    dates = driver.find_elements(By.CLASS_NAME, 'erpl_document-subtitle-fragment')
    dates = [ele.text for ele in driver.find_elements(By.CLASS_NAME, 'erpl_document-subtitle-fragment') if len(ele.text)==10 ]
    urls = driver.find_elements(By.CSS_SELECTOR, 'div.erpl_document-header > h3 > a')

    #titres = driver.find_elements(By.CLASS_NAME, 'erpl_document-header')
    urls = [url.get_attribute('href') for url in urls]

    
    contents = []
    for url in urls :
        driver.get(url)
        time.sleep(np.random.uniform(1,2))
        titre_date = [ele.text for ele in driver.find_elements(By.CSS_SELECTOR, 'td.doc_title')]
        #titres.append(titre)
        content = [ele.text for ele in driver.find_elements(By.CSS_SELECTOR, 'p.contents')]
        #concatenate all the text
        content = " ".join(content)
        contents.append(content)
        
        row = pd.Series([row["name"], titre_date[2], content, titre_date[0]], index=results.columns)
        results.loc[len(results)] = row
        time.sleep(np.random.uniform(1,2))
        #.append({"name": row["name"], "source": titre_date[2], "contenu": content, "source_date":titre_data[0]}, ignore_index=True)
        
    time.sleep(np.random.uniform(2,3))


    

    if i % 3 == 0:
        results.to_csv(f'../data/pleniary_debates/results_{i}.csv', index=False)
    
