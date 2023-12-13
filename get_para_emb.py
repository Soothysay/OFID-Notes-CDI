import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Load data into a pandas dataframe

df1=pd.read_csv('Non_notes.csv')
df2=pd.read_csv('BIJAYA_POS.csv')
df=pd.concat([df1,df2],axis=0)
#df=df1[:200]
#df=df.dropna(subset=['NOTE_TEXT'])
#df.to_csv('data/Grouped_Notes1.csv',index=False)
# Convert dataframe to list of TaggedDocument objects
documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(df['CONCATENATED_NOTES'])]

# Train Doc2Vec model on the corpus
model = model = Doc2Vec(documents, vector_size=128, window=50, min_count=1, workers=10,epochs=500)

# Get embeddings for all notes
embeddings = []
for i in range(len(df)):
    embedding = model.docvecs[i]
    embeddings.append(embedding)

# Create a new dataframe with the embeddings and save to CSV file
embedding_df = pd.DataFrame(embeddings)
embedding_df.to_csv("data/note_embeddings_ex22.csv", index=False)
