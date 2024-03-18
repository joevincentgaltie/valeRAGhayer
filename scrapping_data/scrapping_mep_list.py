import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By

import time
from tqdm import tqdm
import pandas as pd
import json

page_to_scrap = "https://www.europarl.europa.eu/meps/fr/full-list/all"


driver = webdriver.Firefox()
driver.get(page_to_scrap)
time.sleep(2)

#Accept cookies 
driver.find_elements(By.CSS_SELECTOR, 'button.epjs_agree:nth-child(2)')[0].click()
time.sleep(2)

 
meps_blocks=driver.find_elements(By.XPATH, "//*[starts-with(@id, 'member-block')]")

liste_meps = {}
for ele in tqdm(meps_blocks) : 
    description = ele.find_elements(By.XPATH, ".//a")[0].text.split("\n")
    ref = ele.find_elements(By.XPATH, ".//a")[0].get_attribute('href')
    #get number of the mep, after /fr/ 
    number= ref.split("/fr/")[1]
    if len(description) == 4:
        liste_meps[description[0].upper().replace(' ', '_')] = {"party" : description[1], "country" : description[2], "orientation" : description[3], "number": number, "ref" : ref}

    else:

        liste_meps[description[0].upper().replace(' ', '_')] = {"party" : description[1], "country" : description[2], "number": number, "ref" : ref}

pd.DataFrame(liste_meps).T.to_csv("../data/liste_meps.csv")
