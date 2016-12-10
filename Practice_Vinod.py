
# coding: utf-8

# In[155]:

import os


# In[156]:

os.chdir("C:/Users/nakka/Desktop/Python_Jupyter/Practice")


# In[157]:

import numpy as np


# In[158]:

import pandas as pd


# In[159]:

import re


# In[160]:

import matplotlib.pyplot as plt


# In[161]:

data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]


# In[162]:

data = {}


# In[163]:

for f in data_files:
    print("{0}".format(f))
    d = pd.read_csv("{0}".format(f))
    data[f.replace(".csv", "")] = d


# In[164]:

all_survey = pd.read_csv("survey_all.txt", delimiter="\t",encoding='windows-1252')


# In[165]:

d75_survey = pd.read_csv("survey_d75.txt", delimiter="\t",encoding='windows-1252')


# In[166]:

all_survey.shape


# In[167]:

d75_survey.shape


# In[168]:

survey = pd.concat([all_survey, d75_survey], axis=0)


# In[169]:

survey.shape


# In[170]:

data["hs_directory"].head()


# In[171]:

data["sat_results"].head()


# In[172]:

survey["DBN"]=survey["dbn"]


# In[173]:

survey_fields = [
    "DBN", 
    "rr_s", 
    "rr_t", 
    "rr_p", 
    "N_s", 
    "N_t", 
    "N_p", 
    "saf_p_11", 
    "com_p_11", 
    "eng_p_11", 
    "aca_p_11", 
    "saf_t_11", 
    "com_t_11", 
    "eng_t_10", 
    "aca_t_11", 
    "saf_s_11", 
    "com_s_11", 
    "eng_s_11", 
    "aca_s_11", 
    "saf_tot_11", 
    "com_tot_11", 
    "eng_tot_11", 
    "aca_tot_11",
]


# In[174]:

survey = survey.loc[:,survey_fields]


# In[175]:

data["survey"] = survey


# In[176]:

data["survey"].head()


# In[177]:

data["hs_directory"]["DBN"] = data["hs_directory"]["dbn"]


# In[178]:

def pad_csd(num):
    string_representation = str(num)
    if len(string_representation) > 1:
        return string_representation
    else:
        return "0" + string_representation


# In[179]:

data["class_size"].head()


# In[180]:

data["class_size"]["padded_csd"] = data["class_size"]["CSD"].apply(pad_csd)


# In[181]:

data["class_size"].head()


# In[182]:

data["class_size"]["DBN"] = data["class_size"]["padded_csd"] + data["class_size"]["SCHOOL CODE"]


# In[183]:

data["class_size"]["DBN"].head()


# In[184]:

cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']


# In[185]:

for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c], errors="coerce")


# In[186]:

data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]


# In[187]:

data["hs_directory"]["Location 1"].head()


# In[188]:

def find_lat(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lat = coords[0].split(",")[0].replace("(", "")
    return lat


# In[189]:

def find_lon(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lon = coords[0].split(",")[1].replace(")", "").strip()
    return lon


# In[190]:

data["hs_directory"]["lat"] = data["hs_directory"]["Location 1"].apply(find_lat)


# In[191]:

data["hs_directory"]["lon"] = data["hs_directory"]["Location 1"].apply(find_lon)


# In[192]:

data["hs_directory"]["lat"] = pd.to_numeric(data["hs_directory"]["lat"], errors="coerce")


# In[193]:

data["hs_directory"]["lon"] = pd.to_numeric(data["hs_directory"]["lon"], errors="coerce")


# In[194]:

class_size = data["class_size"]


# In[195]:

class_size.shape


# In[196]:

class_size = class_size[class_size["GRADE "] == "09-12"]


# In[197]:

class_size = class_size[class_size["PROGRAM TYPE"] == "GEN ED"]


# In[198]:

data["class_size"]["GRADE "].head()


# In[199]:

class_size = class_size.groupby("DBN").agg(np.mean)


# In[200]:

data["class_size"].head()


# In[201]:

class_size.reset_index(inplace=True)


# In[202]:

data["class_size"] = class_size


# In[203]:

data["demographics"] = data["demographics"][data["demographics"]["schoolyear"] == 20112012]


# In[204]:

data["graduation"] = data["graduation"][data["graduation"]["Cohort"] == "2006"]


# In[205]:

data["graduation"].shape


# In[206]:

data["graduation"] = data["graduation"][data["graduation"]["Cohort"] == "2006"]


# In[207]:

data["graduation"] = data["graduation"][data["graduation"]["Demographic"] == "Total Cohort"]


# In[208]:

data["graduation"].shape


# In[209]:

cols = ['AP Test Takers ', 'Total Exams Taken', 'Number of Exams with scores 3 4 or 5']


# In[210]:

for col in cols:
    data["ap_2010"][col] = pd.to_numeric(data["ap_2010"][col], errors="coerce")


# In[211]:

combined = data["sat_results"]


# In[212]:

combined.shape


# In[213]:

combined = combined.merge(data["ap_2010"], on="DBN", how="left")


# In[214]:

combined.shape


# In[215]:

combined = combined.merge(data["graduation"], on="DBN", how="left")


# In[216]:

combined.shape


# In[217]:

to_merge = ["class_size", "demographics", "survey", "hs_directory"]


# In[218]:

for m in to_merge:
    combined = combined.merge(data[m], on="DBN", how="inner")


# In[219]:

combined.shape


# In[220]:

combined = combined.fillna(combined.mean())
combined = combined.fillna(0)


# In[221]:

def get_first_two_chars(dbn):
    return dbn[0:2]


# In[222]:

combined["school_dist"] = combined["DBN"].apply(get_first_two_chars)


# In[223]:

correlations = combined.corr()


# In[224]:

correlations.sort_values(by="sat_score",ascending = False,inplace = True,na_position="last")


# In[225]:

correlations = correlations["sat_score"]


# In[226]:

correlations.head()


# In[227]:

print(correlations)


# In[228]:

combined.corr()["sat_score"][survey_fields].plot.bar()


# In[229]:

combined.plot.scatter("saf_s_11", "sat_score")


# In[231]:

districts = combined.groupby("school_dist").agg(np.mean)


# In[232]:

districts.reset_index(inplace=True)


# In[234]:

districts.head()


# In[ ]:



