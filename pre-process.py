import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
dfproc = pd.read_csv('files/mimiciii/PROCEDURES_ICD.csv')
dfdiag = pd.read_csv('files/mimiciii/DIAGNOSES_ICD.csv')
dfdiag=dfdiag[['HADM_ID','ICD9_CODE']]
dfdiag.to_csv('diagnoses.csv',index=False)
dfdiag=dfdiag[['HADM_ID','ICD9_CODE']]
dfdiag.to_csv('procedures.csv',index=False)
#!/usr/bin/python3
# Alberto Maria Segre
#
# Copyright 2015, The University of Iowa.  All rights reserved.
# Permission is hereby given to use and reproduce this software 
# for non-profit educational purposes only.
import re
import csv

# Computes the Charelson Comorbidity Index (ccmi), Elixhauser
# Comorbidity vector (ecv) and the vanWalraven/Elixhauser score
# (vwes).

######################################################################
# Charelson Comorbidity disease codes and severity
CDRE = (('^(?:410|412|428)', 1),		# Myocardial infraction
        ('^428', 1),				# Congestive heart failure
        ('^(?:4439|441|7854|v434)', 1),		# Peripheral vascular disease
        ('^43[0-8]', 1), 			# Cerebrovascular disease
        ('^290', 1),				# Dementia
        ('^(?:49[0-9]|50[0-5]|5064)', 1), 	# Chronic pulmonary disease
        ('^(?:710[014]|714[012]|71481|725)', 1),# Rheumatic disease
        ('^(?:53[1-4])', 1), 			# Peptic ulcer disease
        ('^(?:571[24-6])', 1), 			# Mild liver disease
        ('^(?:250[0-37])', 1),			# Diabetes without chronic complication
        ('^(?:250[4-6])', 2), 			# Diabetes with chronic complication
        ('^(?:3441|342)', 2), 			# Hemiplegia or paraplegia
        ('^(?:582|583|583[0-7]?|58[568])', 2),	# Renal disease
        ('^(?:1[4568]|17[124-9]|19[0-4]|195[0-8]|20[0-8]?)', 2), # Maliginancy except of the skin
        ('^(?:456[01]|4562[01]?|572[2-8])', 3),	# Moderate or severe liver disease
        ('^(?:19[678]|199[01]?)', 6), 		# Metastatic solid tumor
        ('^(?:04[2-4])', 6))			# AIDS/HIV
# Charelson Comorbidity procedure codes and severity.
CPRE = (('^3848', 1),)                    	# Peripheral vascular disease

# Elixhauser Comorbidity Vector and van Walraven Comorbidity Score severities.
EDRE = (('^398|402|42[58]', 7, 'chf'), 	# Congestive heart failure
        ('^42[67]', 5, 'car'), 		# Cardiac arrhythmias
        ('^39[4-7]|424|746', -1, 'vvd'),# Valvular disease
        ('^41[5-7]', 4, 'pcd'),		# Pulmonary circulation disorders
        ('^44[0137]|557', 2, 'pvd'),	# Peripheral vascular disorders
        ('^401', 0, 'hyu'),		# Hypertension, uncomplicated
        ('^40[2-5]', 0, 'hyc'),		# Hypertension, complicated
        ('^334|34[2-4]', 7, 'par'),	# Paralysis
        ('^33[1-6]|34[0158]', 6, 'ndd'),	# Neurodegenerative disorders
        ('^416|49[0-6]|50[1-5]', 3, 'cpd'),     # Chronic pulmonary disease
        ('^2500', 0, 'dbu'),	        # Diabetes, uncomplicated
        ('^250[1-9]', 0, 'dbc'),        # Diabetes, complicated
        ('^24[0346]', 0, 'hyt'),	# Hyperthyroidism
        ('^403|58[568]|V56', 5, 'rnf'),	# Renal failure
        ('^070|456|57[0-3]', 11, 'lvd'),# Liver disease
        ('^53[1-4]', 0, 'ppu'),		# Peptic ulcer, no bleeding
        ('^04[2-4]', 0, 'aid'), 	# AIDS
        ('^20[0-3]', 9, 'lym'), 	# Lymphoma
        ('^19[6-9]', 12, 'can'), 	# Metastatic cancer
        ('^1[4568][0-9]|17[0-24-9]|19[0-5]', 4, 'tum'),  # Solid tumor, no metastisis
        ('^446|701|71[0149]|72[058]', 0, 'rar'),         # Rheumatoid arthritis/collagen disease
        ('^28[67]', 3, 'coa'), 		# Coagulopathy
        ('^278', -4, 'obs'), 		# Obesity
        ('^26[1-3]', 6, 'wls'),		# Weight loss
        ('^276', 5, 'fed'),		# Fluid/electrolyte disorders
        ('^2800|2851', -2, 'bla'), 	# Blood loss anemia
        ('^28[01]', -2, 'dfa'), 	# Deficiency anemia
        ('^291|303|980', 0, 'ala'),	# Alcohol abuse
        ('^292|30[45]', -7, 'dra'),	# Drug abuse
        ('^29[3578]', 0, 'psy'),	# Psychosis
        ('^296|30[09]|311', -3, 'dep'))	# Depression

# Compile the regex's.
CDRE = [ (re.compile(tuple[0]), tuple[1]) for tuple in CDRE ]
CPRE = [ (re.compile(tuple[0]), tuple[1]) for tuple in CPRE ]
EDRE = [ (re.compile(tuple[0]), tuple[1], tuple[2]) for tuple in EDRE ]

# Dictionaries for each index.
ccmi = {}
vwes = {}
ecv = {}
# Start with the diagnoses codes.
with open('diagnoses.csv', 'r') as csvfile:
    codes = csv.DictReader(csvfile, dialect='unix')
    vid = None
    for row in codes:
        # Each row will be a dictionary.
        if vid is None:
            vid = row['HADM_ID']
            cmi = 0
            vw = 0
            eh = []
        elif vid != row['HADM_ID']:
            # Save this record
            ccmi[vid] = cmi
            vwes[vid] = vw
            ecv[vid] = eh
            vid = row['HADM_ID']
            cmi = 0
            vw = 0
            eh = []
        # Now process the codes for the Charelson index.
        for tuple in CDRE:
            if tuple[0].match(row['ICD9_CODE']):
                cmi = cmi + tuple[1]
        # And the codes for the Elixhauser vector and the van Walraven
	# score. We could probably do better with the vector, because 
	# this code allows multiple matches from subsequent rows of the 
	# file. That means each 3-letter may occur more than once in 
	# the output for a single visit. But in the long run, it doesn't
	# matter at all, because SQL's SET will ensure only one counts (and
	# having the multiple copies does make debugging this script a 
	# bit easier). 
        for tuple in EDRE:
            if tuple[0].match(row['ICD9_CODE']):
                vw = vw + tuple[1]
                eh.append(tuple[2])
# Handle last iteration.
ccmi[vid] = cmi                
vwes[vid] = vw
ecv[vid] = eh

# Next the procedure codes (only for the Charelson Comorbidity Index).
with open('proceduces.csv', 'r') as csvfile:
    codes = csv.DictReader(csvfile, dialect='unix')
    vid = None
    for row in codes:
        # Each row will be a dictionary.
        if vid is None:
            vid = row['HADM_ID']
        elif vid != row['HADM_ID']:
            # Save this record
            if ccmi[vid] is None:
                ccmi[vid] = cmi
            else:
                ccmi[vid] = ccmi[vid] + cmi
            vid = row['HADM_ID']
            cmi = 0
        # Now process the codes.
        for tuple in CPRE:
            if tuple[0].match(row['ICD9_CODE']):
                  cmi = cmi + tuple[1]
# Handle the last iteration.
if ccmi[vid] is None:
    ccmi[vid] = cmi
else:
    ccmi[vid] = ccmi[vid] + cmi

# Produce output.
print('vid,ccmi,vwes,ecv')
for vid in ccmi.keys():
    print('{},{},{},"{}"'.format(vid, ccmi[vid], vwes[vid], ','.join(ecv[vid])))
items = ccmi.items()
ccmi1=pd.DataFrame({'HADM_ID': [i[0] for i in items], 'ccmi': [i[1] for i in items]})

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
nltk.download('stopwords')
nltk.download('punkt')
dftest = pd.read_csv('files/mimiciii/MICROBIOLOGYEVENTS.csv')
df_CDI=dftest[dftest['ORG_NAME']=='CLOSTRIDIUM DIFFICILE']
df_CDI=df_CDI[['HADM_ID','SUBJECT_ID','CHARTDATE']]
df_patient_notes=pd.read_csv('files/mimiciii/NOTEEVENTS.csv')
hid=df_CDI['HADM_ID'].unique().tolist()
CDI_notes=df_patient_notes.query('HADM_ID == @hid')
CDI_notes=CDI_notes[CDI_notes['CATEGORY']!='Discharge summary']
import re
def preprocess_text(text):
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove dates in the format [**YYYY-MM-DD**]
    text = re.sub(r'\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\]', '', text)
    
    # Remove times in the format [**HH:MM**]
    text = re.sub(r'\[\*\*\d{1,2}:\d{1,2}\*\*\]', '', text)
    
    # Remove demographic information (Date of Birth, Sex)
    text = re.sub(r'Date of Birth:.*?Sex:.*?(?=Service:)', '', text, flags=re.DOTALL)
    
    # Remove newline characters
    text = text.replace('\n', ' ')
    
    # Remove . and -
    text = text.replace('.', ' ')
    text = text.replace('-', ' ')
    
    return text
CDI_notes['TEXT']=CDI_notes['TEXT'].apply(preprocess_text)
CDI_notes=CDI_notes[['HADM_ID','CHARTDATE','TEXT']]
CDI_notes['CHARTDATE']=CDI_notes['CHARTDATE'].astype('datetime64[D]')
df_CDI['CHARTDATE']=df_CDI['CHARTDATE'].astype('datetime64[D]')
from nltk.corpus import wordnet
def synonym_antonym_extractor(phrase):
    synonyms = []

    for syn in wordnet.synsets(phrase):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms
def concatenate_notes(row):
    admission_date = row['CHARTDATE']
    start_date = admission_date - timedelta(days=3)
    filtered_notes = CDI_notes[(CDI_notes['HADM_ID'] == row['HADM_ID']) &
                               (CDI_notes['CHARTDATE'] >= start_date) &
                               (CDI_notes['CHARTDATE'] < admission_date)]
    concatenated_text = "\n".join(filtered_notes['TEXT'])
    #concatenated_text = re.sub(r'\b(?:cdi|cdiff|clostridium difficile|clostridiodes difficile)\b', '', concatenated_text, flags=re.IGNORECASE)
    concatenated_text = re.sub(r'\d+', '', concatenated_text)
    concatenated_text = re.sub(r'\d{1,2}:\d{2}\s*(?:[apAP][mM])?', '', concatenated_text)
    concatenated_text = re.sub(r'\s+', ' ', concatenated_text)  # Remove consecutive spaces
    concatenated_text = re.sub(r'(\: [apAP][mM])', '', concatenated_text)  # Remove ": pm" or ": am"
    concatenated_text = re.sub(r'[_]+', '_', concatenated_text)  # Remove multiple consecutive underscores
    concatenated_text = concatenated_text.lower()
    concatenated_text = re.sub(r'\b(?:cdi|cdiff|clostridium difficile|clostridiodes difficile|c difficile|c diff|cdad|clostridium difficile associated disease|antibiotic associated diarrhea|pseudomembranous colitis|c difficile colitis|c diff colitis|c difficile colonic infection|c difficile enteritis|c difficile associated colitis|c difficile associated enterocolitis|c difficile associated diarrhea|c difficile associated disease|clostridium difficile-associated diarrhea|clostridium difficile colitis|clostridium difficile enteritis|antibiotic associated colitis|aad)\b', '', concatenated_text, flags=re.IGNORECASE)
    
    diarrhea_synonyms1 = synonym_antonym_extractor("diarrhea")
    diarrhea_synonyms = [word.replace('_', ' ') for word in diarrhea_synonyms1]
    stool_synonyms1 = synonym_antonym_extractor("stool")
    stool_synonyms = [word.replace('_', ' ') for word in stool_synonyms1]

    # Function to replace synonyms with root word
    def replace_synonyms(text):
        for word in diarrhea_synonyms:
            text = text.replace(word, "diarrhea")
        for word in stool_synonyms:
            text = text.replace(word, "stool")
        return text

    # Apply the synonym replacement to the concatenated text
    concatenated_text = replace_synonyms(concatenated_text)

    # Remove 'diarrhea' and 'loose stool'
    concatenated_text = re.sub(r'\b(?:diarrhea|loose stool)\b', '', concatenated_text, flags=re.IGNORECASE)
    
    # Tokenize the text
    words = word_tokenize(concatenated_text)
    
    # Remove punctuation and convert to lowercase
    words = [word.lower() for word in words if word.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Join the cleaned words back into a string
    cleaned_text = ' '.join(words)
    if not cleaned_text.strip():
        return None
    return cleaned_text
df_CDI['CONCATENATED_NOTES'] = df_CDI.apply(concatenate_notes, axis=1)
df_CDI=df_CDI.dropna()
df_CDI=df_CDI.reset_index(drop=True)
files=['ABX1.txt','ABX2.txt','ABX3.txt','ABX4.txt','ABX5.txt','GAS1.txt','GAS2.txt']
meds=[]
for file in files:
    wd='pres1/'+file
    loaded_set = set()

    # Open the file in read mode
    with open(wd, 'r') as file1:
        # Read each line from the text file and add it to the set
        for line in file1:
            loaded_set.add(line.strip())  # Strip newline characters and add to the set
    meds.append(loaded_set)
prescriptions=pd.read_csv('files/mimiciii/PRESCRIPTIONS.csv')
prescriptions['ENDDATE']=prescriptions['ENDDATE'].astype('datetime64[D]')
# Step 1: Convert 'DRUG' column in prescriptions DataFrame to lowercase
prescriptions['DRUG'] = prescriptions['DRUG'].str.lower()

# Initialize an empty DataFrame to store the filtered results
filtered_prescriptions = pd.DataFrame()

# Iterate through elements in df_CDI
df_cums=pd.DataFrame(columns=['ABX1_cum','ABX2_cum','ABX3_cum','ABX4_cum','ABX5_cum','GAS1_cum','GAS2_cum'])
print(len(df_CDI))
from tqdm import tqdm
for index, row in tqdm(df_CDI.iterrows()):
    hadm_id = row['HADM_ID']
    chartdate = row['CHARTDATE']- timedelta(days=3)

    # Query prescriptions DataFrame for matching HADM_ID and ENDDATE condition
    filtered_df = prescriptions[(prescriptions['HADM_ID'] == hadm_id) & (prescriptions['ENDDATE']<chartdate)]
    #print(filtered_df)
    # Append the filtered results to the empty DataFrame
    #filtered_prescriptions = pd.concat([filtered_prescriptions, filtered_df])

    # Reset the index of the resulting DataFrame
    filtered_prescriptions.reset_index(drop=True, inplace=True)
    
    cts=[]
    for idx, med_set in enumerate(meds):
        cum_ct=0
        # Convert set elements to lowercase
        med_set = {item.lower() for item in med_set}

        # Initialize a list to store cumulative counts
        cumulative_counts = []

        # Iterate over each row in the filtered DataFrame
        for index, row in filtered_df.iterrows():
            drug = row['DRUG']

            for me in med_set:
                if me in drug:
                    cum_ct+=1
                    break # As 1drug cannot have more than 1 meds
        cts.append(cum_ct)
    df_cums.loc[len(df_cums)]=cts
df_CDI1=pd.concat([df_CDI,df_cums],axis=1)
ccmi1['HADM_ID']=ccmi1['HADM_ID'].map(int)
df_CDI2=pd.merge(df_CDI1, ccmi1, how="left", on=["HADM_ID"])
df_CDI2.dropna()
df_CDI2.to_csv('Positive_CDI_feats.csv',index=False)
df_pats=pd.read_csv('files/mimiciii/PATIENTS.csv')
df_pats=df_pats[['SUBJECT_ID','GENDER','DOB']]
df_ad=pd.read_csv('files/mimiciii/ADMISSIONS.csv')
df_ad=df_ad[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME']]
print(len(df_ad))
merged_df = df_ad.merge(df_pats, on='SUBJECT_ID', how='left')
merged_df=merged_df.dropna()
print(len(merged_df))
df_admissions=merged_df
df_admissions["ADMITTIME"] = df_admissions["ADMITTIME"].astype('datetime64[D]')
df_admissions["DISCHTIME"] = df_admissions["DISCHTIME"].astype('datetime64[D]')
df_admissions["DOB"] = df_admissions["DOB"].astype('datetime64[D]')
#d1=[2097,2099,2100,2101,2102,2103,2104,2105,2106,2107]
#d1=[2138,2139,2140,2141,2142,2143,2144,2145,2146,2147]
df_admissions['AYR']=df_admissions["ADMITTIME"].dt.year
df_admissions['AM']=df_admissions["ADMITTIME"].dt.month
df_admissions['AD']=df_admissions["ADMITTIME"].dt.day
df_admissions['ADAT']=pd.to_datetime(df_admissions["ADMITTIME"]).dt.date
df_admissions['DO']=pd.to_datetime(df_admissions["DOB"]).dt.date
df_admissions['AGE'] = df_admissions.apply(lambda e: (e['ADAT'] - e['DO']).days/365, axis=1).round()
df_admissions=df_admissions.drop_duplicates(subset=['HADM_ID'])
df_full_feats=pd.merge(df_CDI2,df_admissions,on=['HADM_ID','SUBJECT_ID'],how='left')
print(len(df_full_feats))
df_full_feats=df_full_feats.dropna()
print(len(df_full_feats))
df_full_feats=df_full_feats[['CONCATENATED_NOTES', 'ABX1_cum',
       'ABX2_cum', 'ABX3_cum', 'ABX4_cum', 'ABX5_cum', 'GAS1_cum', 'GAS2_cum','AGE','GENDER','ccmi']]
df_full_feats['LABEL']=[1 for i in range(len(df_full_feats))]
df_full_feats.to_csv('BIJAYA_POS.csv',index=False)
df_full_feats=pd.merge(df_CDI2,df_admissions,on=['HADM_ID','SUBJECT_ID'],how='left')
print(len(df_full_feats))
df_full_feats=df_full_feats.dropna()
print(len(df_full_feats))
not_to_include=df_full_feats['SUBJECT_ID'].unique().tolist()
filtered_df = df_admissions[~df_admissions['SUBJECT_ID'].isin(not_to_include)]
hid=filtered_df['HADM_ID'].unique().tolist()
non_notes=df_patient_notes.query('HADM_ID == @hid')
non_notes=non_notes[non_notes['CATEGORY']!='Discharge summary']
hadm=non_notes['HADM_ID'].unique().tolist()
filtered_df=filtered_df.query('HADM_ID == @hadm')
sample=filtered_df.sample(n=9340,random_state=123)

import re

from nltk.corpus import wordnet
def synonym_antonym_extractor(phrase):
    synonyms = []

    for syn in wordnet.synsets(phrase):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

def preprocess_text(text):
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove dates in the format [**YYYY-MM-DD**]
    text = re.sub(r'\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\]', '', text)
    
    # Remove times in the format [**HH:MM**]
    text = re.sub(r'\[\*\*\d{1,2}:\d{1,2}\*\*\]', '', text)
    
    # Remove demographic information (Date of Birth, Sex)
    text = re.sub(r'Date of Birth:.*?Sex:.*?(?=Service:)', '', text, flags=re.DOTALL)
    
    # Remove newline characters
    text = text.replace('\n', ' ')
    
    # Remove . and -
    text = text.replace('.', ' ')
    text = text.replace('-', ' ')
    
    
    
    return text
non_notes['TEXT']=non_notes['TEXT'].apply(preprocess_text)
non_notes=non_notes[['SUBJECT_ID','HADM_ID','TEXT']]
def concatenate_notes(row):
    #admission_date = row['CHARTDATE']
    #start_date = admission_date - timedelta(days=3)
    filtered_notes = non_notes[(non_notes['HADM_ID'] == row['HADM_ID'])]
    concatenated_text = "\n".join(filtered_notes['TEXT'])
    concatenated_text = re.sub(r'\d+', '', concatenated_text)
    concatenated_text = re.sub(r'\d{1,2}:\d{2}\s*(?:[apAP][mM])?', '', concatenated_text)
    concatenated_text = re.sub(r'\s+', ' ', concatenated_text)  # Remove consecutive spaces
    concatenated_text = re.sub(r'(\: [apAP][mM])', '', concatenated_text)  # Remove ": pm" or ": am"
    concatenated_text = re.sub(r'[_]+', '_', concatenated_text)  # Remove multiple consecutive underscores
    concatenated_text = concatenated_text.lower()
    concatenated_text = re.sub(r'\b(?:cdi|cdiff|clostridium difficile|clostridiodes difficile|c difficile|c diff|cdad|clostridium difficile associated disease|antibiotic associated diarrhea|pseudomembranous colitis|c difficile colitis|c diff colitis|c difficile colonic infection|c difficile enteritis|c difficile associated colitis|c difficile associated enterocolitis|c difficile associated diarrhea|c difficile associated disease|clostridium difficile-associated diarrhea|clostridium difficile colitis|clostridium difficile enteritis|antibiotic associated colitis|aad)\b', '', concatenated_text, flags=re.IGNORECASE)
    
    diarrhea_synonyms1 = synonym_antonym_extractor("diarrhea")
    diarrhea_synonyms = [word.replace('_', ' ') for word in diarrhea_synonyms1]
    stool_synonyms1 = synonym_antonym_extractor("stool")
    stool_synonyms = [word.replace('_', ' ') for word in stool_synonyms1]

    # Function to replace synonyms with root word
    def replace_synonyms(text):
        for word in diarrhea_synonyms:
            text = text.replace(word, "diarrhea")
        for word in stool_synonyms:
            text = text.replace(word, "stool")
        return text

    # Apply the synonym replacement to the concatenated text
    concatenated_text = replace_synonyms(concatenated_text)

    # Remove 'diarrhea' and 'loose stool'
    concatenated_text = re.sub(r'\b(?:diarrhea|loose stool)\b', '', concatenated_text, flags=re.IGNORECASE)
    
    # Tokenize the text
    words = word_tokenize(concatenated_text)
    
    # Remove punctuation and convert to lowercase
    words = [word.lower() for word in words if word.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Join the cleaned words back into a string
    cleaned_text = ' '.join(words)
    if not cleaned_text.strip():
        return None
    return cleaned_text
sample['CONCATENATED_NOTES'] = sample.apply(concatenate_notes, axis=1)
sample=sample.dropna()
df_cums=pd.DataFrame(columns=['ABX1_cum','ABX2_cum','ABX3_cum','ABX4_cum','ABX5_cum','GAS1_cum','GAS2_cum'])
sample1=sample.reset_index(drop=True)
print(len(sample))
from tqdm import tqdm


for index, row in tqdm(sample1.iterrows()):
    hadm_id = row['HADM_ID']
    #chartdate = row['CHARTDATE']- timedelta(days=3)

    # Query prescriptions DataFrame for matching HADM_ID and ENDDATE condition
    filtered_df = prescriptions[(prescriptions['HADM_ID'] == hadm_id)]
    #print(filtered_df)
    # Append the filtered results to the empty DataFrame
    #filtered_prescriptions = pd.concat([filtered_prescriptions, filtered_df])

    # Reset the index of the resulting DataFrame
    filtered_prescriptions.reset_index(drop=True, inplace=True)
    
    cts=[]
    for idx, med_set in enumerate(meds):
        cum_ct=0
        # Convert set elements to lowercase
        med_set = {item.lower() for item in med_set}

        # Initialize a list to store cumulative counts
        cumulative_counts = []

        # Iterate over each row in the filtered DataFrame
        for index, row in filtered_df.iterrows():
            drug = row['DRUG']

            for me in med_set:
                if me in drug:
                    cum_ct+=1
                    break # As 1drug cannot have more than 1 meds
        cts.append(cum_ct)
    df_cums.loc[len(df_cums)]=cts
sample2=pd.concat([sample1,df_cums],axis=1)
sample2=pd.merge(sample2, ccmi1, how="left", on=["HADM_ID"])
sample2=sample2[['CONCATENATED_NOTES', 'ABX1_cum',
       'ABX2_cum', 'ABX3_cum', 'ABX4_cum', 'ABX5_cum', 'GAS1_cum', 'GAS2_cum','AGE','GENDER','ccmi']]
sample2['LABEL']=[0 for i in range(len(sample))]
sample2.to_csv('Non_notes.csv',index=False)