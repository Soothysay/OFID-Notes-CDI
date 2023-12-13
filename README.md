# ICHE-Notes-CDI

Please reach out to akash-choudhuri@uiowa.edu for any queries.

## Setting up:
1. Please set up your environment using the environment.yml file
2. Please get access to MIMIC-III through the credentialing process. Details are in https://mimic.mit.edu/
3. Store your data locally. In our case our data was stored in the directory files/mimiciii/

## Usage:
1. Please modify the file paths according to your own locations to pre-process and get the case files and run python3 pre-process.py
2. Get the Doc2vec embeddings of the created case files by running python3 get_para_emb.py
3. Get the results of the Logistic Regression Models by running python3 LR_exp_vars.py
4. To get t-SNE visualization of the note embeddings run python3 TSNE_tunder.py
