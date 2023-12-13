import pandas as pd
from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
import numpy as np
d=pd.read_csv('data/note_embeddings_ex22.csv')
df1=pd.read_csv('Non_notes.csv')
df2=pd.read_csv('BIJAYA_POS.csv')
df=pd.concat([df1,df2],axis=0)
df=df.reset_index(drop=True)
print(len(df))
print(len(d))
df=df[['LABEL']]
df3=pd.concat([df,d],axis=1)
df3=df3.reset_index(drop=True)
df3.columns.tolist()
dfx=df3.drop(['LABEL'],axis=1)

for i in range(1,600):
    import matplotlib.pyplot as plt
    loc='TSNE/TSNE_plot'+str(i)+'.png'
    pca = TSNE(n_components=2,n_iter=5000, learning_rate='auto',init='random', perplexity=i)
    vals=pca.fit_transform(dfx.values)
    colu=df3['LABEL'].tolist()
    plt.figure(figsize=(10, 6))
    scatter=plt.scatter(vals[:,0], vals[:,1],c=colu, alpha=0.5)
    # Create a legend with the specified labels
    #legend = plt.legend(handles=[scatter], title='Legend', loc='best')  # 'handles' specifies the objects to include in the legend
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('TSNE Result')
    plt.grid()
    plt.savefig(loc)
    