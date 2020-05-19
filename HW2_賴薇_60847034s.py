#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:17:49 2020

@author: weilai
"""
import os
import re
import pandas as pd
import math
import numpy as np
#==============================================================================
def ReadPath(path):
    count = 0
    dir_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            #print(os.path.join(root, f))
            dir_files.append(os.path.join(root, f))
            count += 1
    return count, dir_files
#==============================================================================
def ReadFile(dirs):
    text_list = []
    for i in range(len(dirs)):
        try:
            f = open(dirs[i], 'r')
            temp_text = []
            for line in f:
                 temp_text.append([line])
        except OSError:
             print('Cannot open ', dirs[i])    
        text_list.append(temp_text)
    return text_list
#==============================================================================
def ReadMaxTerm(text):
    max = 0
    for i in range(len(text)):
        for j in range(3, len(text[i])):
            #print(text[i][j])
            text[i][j] = re.sub(r'-1', '', str(text[i][j]))
            #print(str(text[i][j]))
            num_list = re.findall(r'[0-9]+', text[i][j]) 
            #print(num_list)
            for k in range(len(num_list)):
                if int(num_list[k]) > max:
                    max = int(num_list[k])
    return max
#==============================================================================
def GenerateDocumentVector(text, df_frequency):
    for i in range(len(text)):
        # For each doc generate their vector
        #print(text[i])
        for j in range(3, len(text[i])):
            text[i][j] = re.sub(r'-1', '', str(text[i][j]))
            #print(str(text[i][j]))
            num_list = re.findall(r'[0-9]+', text[i][j]) 
            #print(num_list)
            for k in range(len(num_list)):
                #print(i)
                #print('before', df_frequency.iloc[int(num_list[k]), i])
                df_frequency.iloc[int(num_list[k]), i] += 1
                #print('after', df_frequency.iloc[int(num_list[k]), i])
    return df_frequency
#==============================================================================
def GenerateQueryVector(text, df_query):
    for i in range(len(text)):
        # For each doc generate their vector
        #print(text[i])
        for j in range(len(text[i])):
            text[i][j] = re.sub(r'-1', '', str(text[i][j]))
            #print(str(text[i][j]))
            num_list = re.findall(r'[0-9]+', text[i][j]) 
            #print(num_list)
            for k in range(len(num_list)):
                df_query.iloc[int(num_list[k]), i] += 1
    return df_query
#==============================================================================
def Initial_Document_DF(index, columns, text_list, docs_count):
    df = pd.DataFrame(index=index, columns=columns)
    df = df.fillna(0)
    df = GenerateDocumentVector(text_list, df)
    df.to_csv("docs_frequency.csv")
    
    term_appear_in_text = []
    for i in range(df.shape[0]):
        times = 0
        for j in range(df.shape[1]):
            if df.iloc[i, j] > 0:
                times += 1
        term_appear_in_text.append(times)
  
    df['appear_times'] = term_appear_in_text
    #df.to_csv("docs_frequency_with_appear_times.csv")
  
    idf = []
    for i in range(df.shape[0]):
        temp = 0
        if df['appear_times'][i] > 0:
            temp = math.log(docs_count/df['appear_times'][i])
        idf.append(temp)
        
    df['IDF'] = idf
    df.to_csv("docs_frequency_with_IDF.csv")
#==============================================================================
def Initial_Query_DF(index, columns, text_list, q_count):
    df = pd.DataFrame(index=index, columns=columns)
    df = df.fillna(0)
    df = GenerateQueryVector(text_list, df)
    df.to_csv("q_frequency.csv")
    
    term_appear_in_text = []
    for i in range(df.shape[0]):
        times = 0
        for j in range(df.shape[1]):
            if df.iloc[i, j] > 0:
                times += 1
        term_appear_in_text.append(times)
  
    df['appear_times'] =  term_appear_in_text
    df.to_csv("q_frequency_with_appear_times.csv")
#==============================================================================
def Build_Docs_Representation(df, colname):
    for i in range(len(colname)):
        #print(colname[i])
        #print(df.shape[0])
        for j in range(df.shape[0]):
            #print(df.iloc[j, i])
            if df.iloc[j, i] > 0:
                #print(df.iloc[j, i])
                df.iloc[j, i] = (1 + math.log(df.iloc[j, i])) * df.loc[j, 'IDF']
    df.to_csv("docs_representation.csv")
#==============================================================================
def Build_Query_Representation(df, colname, df_IDF):
    for i in range(len(colname)):
        #print(colname[i])
        #print(df.shape[0])
        for j in range(df.shape[0]):
            #print(df.iloc[j, i])
            if df.iloc[j, i] > 0:
                #print(df.iloc[j, i]) 
                print(df_IDF.loc[j])
                df.iloc[j, i] = (1 + math.log(df.iloc[j, i])) * df_IDF.loc[j]
    df.to_csv("query_representation.csv")
#============================================================================== 
def Length_of_Vector(arr, colname):
    length = []
    
    for i in range(len(colname)):
        # Calculate length of docs_i
        len_doc_i = 0
        for j in range(arr.shape[0]):
            #print('i:', i)
            #print('j', j)
            if arr[j, i] > 0:
                len_doc_i += arr[j, i] * arr[j, i]
        len_doc_i = math.sqrt(len_doc_i)
        #print(len_doc_i)
        length.append(len_doc_i)
        
    return length
#==============================================================================
def Similarity(arr_docs, colname_docs, arr_q, colname_q):
    print(arr_docs)
    print(arr_q)
    arr_cosine_similarity = np.zeros((arr_docs.shape[1], arr_q.shape[1]), dtype = float)
    print(arr_cosine_similarity)
    print(arr_cosine_similarity.shape)
   
    length_docs = Length_of_Vector(arr_docs, colname_docs)
    print(length_docs)
    length_q = Length_of_Vector(arr_q, colname_q)
    print(length_q)
    
    # For each query, caluculate its similarity with doc_i
    for i in range(len(colname_docs)):
        for j in range(len(colname_q)):
            # Calculate similarity
            temp = 0
            for k in range(arr_docs.shape[0]):
                if arr_docs[k, i] > 0 and arr_q[k, j] > 0:
                    temp += arr_docs[k, i] * arr_q[k, j]
            temp = temp / (length_docs[i] * length_q[j])
            print('i:', i, 'j:', j)
            arr_cosine_similarity[i, j] = temp
    np.savetxt("similarity.csv", arr_cosine_similarity, delimiter=",")
#==============================================================================
if __name__ == '__main__':
    
    # Read Documents
    path = './SPLIT_DOC_WDID_NEW'
    docs_count, dir_docs = ReadPath(path)
    print(docs_count)
    #print(dir_docs)
    docs_text_list = ReadFile(dir_docs)
    #print(docs_text_list[0])
    max_term_size = 0
    max_term_size = ReadMaxTerm(docs_text_list)
    #print(max_term_size)
    index_docs = [i for i in range(max_term_size+1)]
    columns_docs = [dir_docs[i][21:] for i in range(len(dir_docs))]
    #print(columns)
    
    
    # Initial Documents
    Initial_Document_DF(index_docs, columns_docs, docs_text_list, docs_count)
   

    # Read Queries
    path= './QUERY_WDID_NEW'
    q_count, dir_q = ReadPath(path)
    query_text_list = ReadFile(dir_q)
    index_query = [i for i in range(max_term_size+1)]
    columns_query = [dir_q[i][17:] for i in range(len(dir_q))]
    print(columns_query)
    
    
    # Initial Query
    Initial_Query_DF(index_query, columns_query, query_text_list, q_count)
    
    
    
    # Build doocument Representation
    df_document = pd.read_csv('docs_frequency_with_IDF.csv', low_memory = False)
    df_document = df_document.drop(['Unnamed: 0'], axis=1)
    Build_Docs_Representation(df_document, columns_docs)
    
    # Build query Representation
    df_query = pd.read_csv('q_frequency_with_appear_times.csv', low_memory = False)
    df_query = df_query.drop(['Unnamed: 0'], axis=1)
    df_document = pd.read_csv('docs_frequency_with_IDF.csv', low_memory = False)
    df_document = df_document.drop(['Unnamed: 0'], axis=1)
    Build_Query_Representation(df_query, columns_query, df_document['IDF'])
   
    # Calculate similarity
    df_document = pd.read_csv('docs_representation.csv', low_memory = False)
    df_document = df_document.drop(['Unnamed: 0'], axis=1)
    df_document = df_document.drop(['appear_times'], axis=1)
    df_IDF = df_document['IDF']
    df_document = df_document.drop(['IDF'], axis=1)
    
    arr_doc = df_document.to_numpy()
    print(arr_doc)
    print(arr_doc.shape)
    
    df_query = pd.read_csv('query_representation.csv', low_memory = False)
    df_query = df_query.drop(['Unnamed: 0'], axis=1)
    df_query = df_query.drop(['appear_times'], axis=1)
    
    arr_q = df_query.to_numpy()
    print(arr_q)  
    print(arr_q.shape)
    
    Similarity(arr_doc, columns_docs, arr_q, columns_query)
    
    
    similarity_arr = np.loadtxt("similarity.csv", dtype=np.float, delimiter=',')
    #print(similarity_arr)
    
    
    df_doc_name = pd.DataFrame(columns_docs)
    #print(df_doc_name)
    final = pd.DataFrame(columns = ['file', 'rating'])
    
    for i in range(similarity_arr.shape[1]):
        df_s = pd.DataFrame(similarity_arr[:, i])
        
        result = pd.concat([df_doc_name, df_s], axis=1, sort = False)
        #print(result)
        result.columns = ['file', 'rating']
        result = result.sort_values(by=['rating'], ascending = False)
        #result = result[result['rating']>0]
    
        final = pd.concat([final, result], axis = 0, sort = False)
        #print(final)
        result.to_csv(columns_query[i]+'.txt', header = False, index = False, sep=" ")
        
    
        