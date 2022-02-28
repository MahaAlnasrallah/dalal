# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 08:32:32 2022

@author: maha alnasrallah
"""
import pandas as pd
import re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')  
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')    
    
    
def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=True)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    
def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]])) 

def getclusters(sentiment_flag=0,no_clusters=10,no_words=10):
    data = pd.read_excel(r'C:\Users\96654\twee3\BERT_tweets_RESULTS_9_2.xlsx')
 
    data=data[data['sentiment_BERT'] == sentiment_flag]
    #data=data[data['Search_word'] == "الضريبة الانتقائية"]


    #remove english words and characters
    data['content'] = data['content'].replace('\d+', ' ',regex=True)
    data['content'] = data['content'].str.replace(r'\n', ' ', regex=True)
    data['content'] = data['content'].str.replace(r'http\S+', ' ', regex=True)
    data['content'] = data['content'].str.replace(r"[a-zA-Z0-9]+", ' ', regex=True)
    data['content'] = data['content'].str.replace(r'\W', ' ', regex=True)
    data['content'] = data['content'].str.replace(r'_', ' ', regex=True)
    #remove keywords search
    key_words = pd.read_excel(r'C:\Users\96654\keys2.xlsx')
    key_words = key_words['KEY'].tolist()
    ho=data['content']
    for special_char in key_words:
        ho=ho.replace(special_char, ' ',regex=True)
        
        special_char_list = ['ضريبه','والجمارك','العارية','التصرفات','المضافه','والمضافه','والضريبة','الضريبة','الضريبه','المضافة','الفاتورة','الضرائب','والجمارك','الزكاة','الزكاه','والدخل','القيمة','الإنتقائية','المضافة','الضرائب','القيمه']
    for special_char in special_char_list:
        ho=ho.replace(" "+special_char+" ", ' ',regex=True)
        
        stop_words = ['لها ','مثل ','علي ','الي ','له ','قيمة','إقرار','الإقرار','اذا ','هل ','ما ','الى ','هي ','ولا ','هذا ','لكن ','والدخل','لا ','ولا ','الإنتقائية','لم ','إلى ','هو ','انها ','أو ','كل ','ذلك ','كان ','حتى ','ال ','التي ','هذه ','أن ','ان ','كانت ','بس ','هذي ','منها ',
    'الزكاه','الزكاة','ضريبة',' مع ','من ','على ','هـ ','عن ','عند ','بما ','او ','بأنة ','وقد ','الإ ','بان ',' بانه'
    ,'في ','عنها ','عند ','عليها ','انه ','عليه ','بإن ','أيضا ','عنه ','أيضا ','وبعدها ','بأن ','بعد ','أن ','أنه ','إنه ', 'تم ']
    for special_char in stop_words:
        ho=ho.replace(" "+special_char, ' ',regex=True)
        
    data['content']=ho
    
    
    tfidf = TfidfVectorizer(
        min_df = 5,
        max_df = 0.95,
        max_features = 8000,
    )
    tfidf.fit(data.content)
    text_tfidf = tfidf.fit_transform(data.content)
    
    cv = CountVectorizer()
    text_cv = cv.fit_transform(data.content)
    
    
        
    find_optimal_clusters(text_tfidf, 50)
    
    
    clusters = MiniBatchKMeans(n_clusters=no_clusters , init_size=1024, batch_size=2048, random_state=20).fit_predict(text_tfidf)
    
    plot_tsne_pca(text_tfidf, clusters)


            
    return get_top_keywords(text_tfidf, clusters, tfidf.get_feature_names(), no_words)

