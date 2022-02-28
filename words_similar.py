# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:57:39 2022

@author: maha alnasrallah
"""
import numpy as np
from scipy import linalg
import pandas as pd
import re
import scipy.stats as stats
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt 


def cosine_similarity(vector_1: np.ndarray, vector_2: np.ndarray) -> float:

    return np.dot(vector_1, vector_2)/(np.linalg.norm(vector_1) *
                                          np.linalg.norm(vector_2))


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:

    return np.linalg.norm(x - y)




def getWordSimilarity(Similars,Not_Similars,sentiment_flag=0,k=100,cos=0.50):

    d3 = pd.read_excel(r'C:\Users\96654\twee3\BERT_tweets_RESULTS_9_2.xlsx')

    d3['content'] = d3['content'].replace('\d+', ' ',regex=True)
    d3['content'] = d3['content'].str.replace(r'\n', ' ', regex=True)
    d3['content'] = d3['content'].str.replace(r'http\S+', ' ', regex=True)
    d3['content'] = d3['content'].str.replace(r"[a-zA-Z0-9]+", ' ', regex=True)
    d3['content'] = d3['content'].str.replace(r'\W', ' ', regex=True)
    d3['content'] = d3['content'].str.replace(r'_', ' ', regex=True)
    ho=d3['content']

        
    stop_words = ['لها ','مثل ','علي ','الي ','له ','اذا ','هل ','ما ','الى ','هي ','ولا ','هذا ','لكن ','لا ','ولا ','لم ','إلى ','هو ','انها '
                  ,'أو ','كل ','ذلك ','كان ','حتى ','ال ','التي ','هذه ','أن ','ان ','كانت ','بس ','هذي ','منها '
                 ,' مع ','من ','على ','هـ ','عن ','عند ','بما ','او ','بأنة ','وقد ','الإ ','بان ',' بانه'
    ,'في ','عنها ','عند ','عليها ','انه ','عليه ','بإن ','أيضا ','عنه ','أيضا ','وبعدها ','بأن ','بعد ','أن ','أنه ','إنه ', 'تم ']
    for special_char in stop_words:
        ho=ho.replace(" "+special_char, ' ',regex=True)
        
    d3['content']=ho
    
   
    
    
    
    
    d3=d3[d3['sentiment_BERT'] == sentiment_flag]


    
    
    cv = CountVectorizer()
    X = np.array(cv.fit_transform(d3['content'].tolist()).todense())
    df = pd.DataFrame(data=X,columns = cv.get_feature_names())
    df_T= df.T
    df_T.reset_index(inplace=True)
    df_T = df_T.rename(columns = {'index':'words'})
    df_T= df_T[['words']]
    
    
    Z_Score = stats.zscore(X,  axis=0)
    
    U, s, VT = linalg.svd(Z_Score.T,full_matrices=False)
    reduce_matrix_x =(U[:, :400]) * np.sqrt(s[:400])
    
    plt.plot(s)
    
    Ind_Similars=[]
    try:
    #Similars =['الاحتكار','ضوابط','العقار']
        Ind_Similars = [df_T.words.tolist().index(i) for i in Similars]
    except ValueError:
        print( "No words for similars in the dictionary")  
        return 
    if Ind_Similars:
        print( "similar words:\n" ,df_T.words[Ind_Similars])
    else:
        print("No Match indices")
        
        
    Vect_Similars=reduce_matrix_x[Ind_Similars] 
    Vect_Similars=0
    for Ind in Ind_Similars:
        Vect_Similars=Vect_Similars+reduce_matrix_x[Ind]
    Vect_Similars 
        
    
    Ind_Not_Similars=[]
    try:
    #Not_Similars =['البورصة']
        Ind_Not_Similars = [df_T.words.tolist().index(i) for i in Not_Similars]
    except ValueError:
        print( "No words for not-similars in the dictionary")  
        return 
           
    if Ind_Not_Similars:
    
        print( "not similar words:\n" ,df_T.words[Ind_Not_Similars])
    else:
        print("No Match indices for not similar words")
        
        
        
    Vect_Not_Similars=0
    for Ind in Ind_Not_Similars:
        Vect_Not_Similars=Vect_Not_Similars+reduce_matrix_x[Ind]
    Vect_Not_Similars
    
    
    
    v =Vect_Similars-Vect_Not_Similars
    candidates = reduce_matrix_x
    
    sorted_list = sorted(range(len(candidates)), key=lambda x: ( euclidean_distance(v, candidates[x])), reverse=False)
    
    sorted_list_df= pd.DataFrame(sorted_list[:k])
    
    a_list = [cosine_similarity(v, row) for row in candidates[sorted_list[:k]]]
    cosin = [idx for idx, element in enumerate(a_list) if element>cos]
    cosin_df = pd.DataFrame(cosin)
    
    
    a=sorted_list[:k]
    T = [a[i] for i in cosin]
    T = pd.DataFrame(T)
    
    merge = sorted_list_df.merge(cosin_df, left_on=0 ,right_on=0)
    merge2= T.merge(df_T, left_on=0 ,right_index=True)

  
    return merge2

    return merge2















