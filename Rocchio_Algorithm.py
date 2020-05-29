#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:54:49 2020

@author: weilai
"""
#==============================================================================
import pandas as pd
import numpy as np
import math
import re 
from scipy.spatial import distance
#==============================================================================
def Read_File(f):
    text = []
    temp = []
    query_amount = []
    Count = 0
    for line in f:
        if 'Query' not in str(line) and 'VOM' in str(line):
            temp.extend([line])
            Count+=1
        elif 'Query' in str(line):
            amount = re.findall(r'query[ 0-9]+', str(line))
            amount = re.sub(r'[A-z ]+', '', str(amount[0]))
            query_amount.append(int(amount))
        if Count == int(amount):
            text.extend([temp])
            temp = []
            Count = 0
    return text, query_amount
#============================================================================== 
def Length_of_Vector(arr, colname):
    length = []
    
    for i in range(colname):
        # Calculate length of docs_i
        len_doc_i = 0
        for j in range(arr.shape[0]):
            if arr[j, i] > 0:
                len_doc_i += arr[j, i] * arr[j, i]
        len_doc_i = math.sqrt(len_doc_i)
        length.append(len_doc_i)
        
    return length
#============================================================================== 
def Similarity(arr_docs, colname_docs, arr_q, colname_q):
    print(arr_docs)
    print(arr_docs.shape)
    print(arr_q)
    print(arr_q.shape)
    arr_cosine_similarity = np.zeros((arr_docs.shape[1], arr_q.shape[1]), dtype = np.float64)
    
    print(arr_cosine_similarity)
    print(arr_cosine_similarity.shape)
   
    length_docs = Length_of_Vector(arr_docs, colname_docs)
    print(length_docs)
    length_q = Length_of_Vector(arr_q, colname_q)
    print(length_q)
   
    # For each query, caluculate its similarity with doc_i
    for i in range(colname_docs):
        for j in range(colname_q):
            # Calculate similarity
            temp = 0
            for k in range(arr_docs.shape[0]):
                if arr_docs[k, i] > 0 and arr_q[k, j] > 0:
                    temp += arr_docs[k, i] * arr_q[k, j]
            temp = temp / (length_docs[i] * length_q[j])
            #temp = distance.cosine(arr_docs[:, i], arr_q[:, j])
            arr_cosine_similarity[i, j] = temp
    np.savetxt("similarity_new.csv", arr_cosine_similarity, delimiter=",")
#==============================================================================
if __name__ == '__main__':
   
    df_document = pd.read_csv('docs_representation.csv', low_memory = False)
    df_document = df_document.drop(['appear_times'], axis=1)
    df_document = df_document.drop(['IDF'], axis=1)
    
    
    Q_related = np.zeros((51249, 16), dtype = np.float64)
    Q_unrelated = np.zeros((51249, 16), dtype = np.float64)
    Q_new = np.zeros((51249, 16), dtype = np.float64)
    Sum = np.zeros((51249, ), dtype = np.float64)
    
    # Read AssessmentTrainSet.txt
    try:
        f_assessment = open(r'./AssessmentTrainSet.txt', 'r')
        text_assesssment, query_assesment_size = Read_File(f_assessment)
        #print(len(text_assesssment))
        #print(text_assesssment)
        #print(query_assesment_size)
        for i in range(len(text_assesssment)):
            for j in range(len(text_assesssment[i])):
                text_assesssment[i][j] = text_assesssment[i][j][:-1] 
                print(text_assesssment[i][j])
                Q_related[:, i] = Q_related[:, i]+df_document[str(text_assesssment[i][j])].to_numpy()
            print(Q_related[:, i])
    except OSError:
         print('Cannot open AssessmentTrainSet.txt')
    
    
    np.savetxt("Q_related.csv", Q_related, delimiter=",")

    df_document = df_document.drop(['Unnamed: 0'], axis=1)
    columns = df_document.columns
    print(columns)
  
    doc_representation = df_document.to_numpy()
    Sum = np.sum(doc_representation, axis=1)
    np.savetxt("Sum.csv", Sum, delimiter=",")
    
    for i in range(16):
        Q_unrelated[:, i] = Sum - Q_related[:, i] 
    
    
    df_query = pd.read_csv('query_representation.csv', low_memory = False)
    df_query = df_query.drop(['Unnamed: 0'], axis=1)
    df_query = df_query.drop(['appear_times'], axis=1)
    columns_query = df_query.columns
    Q_new = df_query.to_numpy()
    
    for i in range(16):
        print(len(text_assesssment[i]))
        #Q_new[:, i] = Q_new[:, i]+0.75/len(text_assesssment[i]) * Q_related[:, i] - (0.25/(2265 - len(text_assesssment[i]))) * Q_unrelated[:, i]
        Q_new[:, i] = Q_new[:, i] + Q_related[:, i] * 0.8/len(text_assesssment[i]) 
        #Q_new[:, i] = np.where(Q_new[:, i]>0, Q_new[:, i], 0)
    #np.savetxt("query_representation_new.csv", Q_new, delimiter=",")
    
    
    # Step 2
    #Similarity(doc_representation, 2265, Q_new, 16)
 
    # Step 3
    similarity_arr = np.loadtxt("similarity_new.csv", dtype=np.float64, delimiter=',')
    
    df_doc_name = pd.DataFrame(columns)
    
    f_result =  open('./ResultsTrainSet.txt', "w",encoding='utf-8')

    for i in range(similarity_arr.shape[1]):
        f_result.write('\nQuery '+str(i+1)+'      '+str(columns_query[i])+' 2265\n')
        
        df_s = pd.DataFrame(similarity_arr[:, i])
        result = pd.concat([df_doc_name, df_s], axis=1, sort = False)
        result.columns = ['file', 'rating']
        result = result.sort_values(by=['rating'], ascending = False)
        for j in range(result.shape[0]):
            f_result.write(str(result.iloc[j, 0])+' '+str(result.iloc[j, 1])+'\n')
        #result.to_csv(columns_query[i]+'.txt', header = False, index = False, sep=" ")
    
    f_result.close()   
   

   