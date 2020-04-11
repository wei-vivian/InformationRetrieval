import re
import numpy as np
import matplotlib.pyplot as plt
import math
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
def Query_Size(text1, text2):
    q_size = -1
    if len(text1) == len(text2):
        q_size = len(text1)
    return q_size
#==============================================================================
def Calculate_Precision_Recall(text1, text1_size, text2, text2_size):
    # Calculate precesion and recall for each document
    np_Arr = np.zeros((text2_size, 2), dtype = 'f')
    count = 0
    correct = 0
    for i in range(text2_size):
        count += 1
        for j in range(text1_size):
            if str(text1[j][:-1]) in str(text2[i]):
                correct += 1
                break
        p = correct/count
        r = correct/int(str(text1_size))
        
        np_Arr[count-1][0] = p
        np_Arr[count-1][1] = r
    return np_Arr
#==============================================================================
def Interpolated_Recall_Precision_Curve(text1, text1_size, text2, text2_size):
    
    query_size = Query_Size(text1, text2)
    if query_size == -1:
        print('Query amounts are not the same betwwen text1 and text2.')
    
    # Evaluation
    np_Arr_final = np.zeros(11, dtype = 'f')
    for i in range(query_size):

        # Calculate precision and recall
        np_Arr = np.zeros((text2_size[i], 2), dtype = 'f')
        np_Arr = Calculate_Precision_Recall(text1[i], text1_size[i], text2[i], text2_size[i])
        
        # Interpolated
        np_Arr2 = np.zeros(11, dtype = 'f') 
        k = 0.1
        k_index = 0
        for j in range(text2_size[i]):
            if np_Arr[j][1] < k and k<1.1:
                if np_Arr2[k_index] == 0 or np_Arr[j][0] > np_Arr2[k_index]:
                    np_Arr2[k_index] = np_Arr[j][0]
            else:
                k += 0.1
                k_index+=1
                
        # Fill empty value        
        for j in range(11):
            if np_Arr2[j] == 0:
                for k in range(j+1, 11, 1):
                    if np_Arr2[k] != 0:
                        np_Arr2[j] = np_Arr2[k]
                        break
        
        # Sum the overall precision and recall in this time step
        np_Arr_final = np_Arr_final+np_Arr2
    
    try:
        np_Arr_final = np_Arr_final/query_size
    except ZeroDivisionError as devide_by_zero:
        print(devide_by_zero)
        
    #print(np_Arr_finall)
    
    interpolated_recall = np.zeros(11, dtype = 'f')
    for i in range(11):
        interpolated_recall[i] = 0.1*i
    
    # plot Interpolated Recall-Precision Curve
    plt.style.use('seaborn-whitegrid')
    plt.plot(interpolated_recall, np_Arr_final)
    plt.xlim(0,1)
    plt.ylim(0,1)
    my_x_ticks = np.arange(0, 1.1, 0.1)
    my_y_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Interpolated Recall-Precision Curve')
    plt.savefig('HW1_Interpolated Recall-Precision Curve.png')
    #plt.show()
#==============================================================================
def Mean_Average_Precision(text1, text1_size, text2, text2_size):
    
    query_size = Query_Size(text1, text2)
    if query_size == -1:
        print('Query amounts are not the same betwwen text1 and text2.')
        
    # Evaluation
    np_Arr_final = np.zeros(query_size, dtype = 'f')
    for i in range(query_size):
        
        # Calculate precision and recall
        np_Arr = np.zeros((text2_size[i], 2), dtype = 'f')
        np_Arr = Calculate_Precision_Recall(text1[i], text1_size[i], text2[i], text2_size[i])
        
        current_recall = -1
        sum_precesion = 0
        sum_precesion_count = 0
        for j in range(text2_size[i]):
            if np_Arr[i][1] > current_recall:
                sum_precesion = np_Arr[i][0]
                sum_precesion_count += 1
                current_recall = np_Arr[i][1]
        try:
            np_Arr_final[i] = sum_precesion / sum_precesion_count
        except ZeroDivisionError as devide_by_zero:
            print(devide_by_zero)
            
    sum_precsion_all_queries = 0
    for i in range(query_size):
        sum_precsion_all_queries += np_Arr_final[i]
    avg_precsion_all_queries = 0
    try:
        avg_precsion_all_queries = sum_precsion_all_queries / query_size
    except ZeroDivisionError as devide_by_zero:
            print(devide_by_zero)
    
    print('Mean Average Precision: ', avg_precsion_all_queries)
#==============================================================================
def Cumulated_Gain(Arr, size):
    CG_Arr = np.zeros(size, dtype = 'f')
    count = 0
    for i in range(size):
        count += Arr[i]
        CG_Arr[i] = count
    #print(CG_Arr)
    return CG_Arr
#==============================================================================
def Discounted_Cumulated_Gain(Arr, size):
    DCG_Arr = np.zeros(size, dtype = 'f')
    for i in range(size):
        if i == 0:
            DCG_Arr[i] = Arr[i]
        else:
            DCG_Arr[i] = Arr[i] / math.log2(i+1) + DCG_Arr[i-1]
    #print(DCG_Arr)
    return DCG_Arr
#==============================================================================
def Normalized_DCG(text1, text1_size, text2, text2_size):
    
    # Query size
    query_size = Query_Size(text1, text2)
    if query_size == -1:
        print('Query amounts are not the same betwwen text1 and text2.')
    
    max_document_size = 0
    for i in range(query_size):
        if text2_size[i] > max_document_size: 
            max_document_size += text2_size[i]
    #print(max_document_size)
    
    sum_DCG_Arr = np.zeros(max_document_size, dtype = 'f')
    avg_DCG_Arr = np.zeros(max_document_size, dtype = 'f')
    sum_IDCG_Arr = np.zeros(max_document_size, dtype = 'f')
    avg_IDCG_Arr = np.zeros(max_document_size, dtype = 'f')
    
    for i in range(query_size):

        # Record whether text2[i][j] exists in text1[i][k]
        text2_match_text1_Arr = np.zeros(text2_size[i], dtype = 'f')
        for j in range(text2_size[i]):
            for k in range(text1_size[i]):
                if str(text1[i][k][0:-1]) in str(text2[i][j]):
                    text2_match_text1_Arr[j] = 1
                    break

        # CG: Cumulated ranking of all document in Query[i] 
        CG_Arr = np.zeros(text2_size[i], dtype = 'f')
        CG_Arr = Cumulated_Gain(text2_match_text1_Arr, text2_size[i])
        
        # DCG
        DCG_Arr = np.zeros(text2_size[i], dtype = 'f')
        DCG_Arr = Discounted_Cumulated_Gain(CG_Arr, text2_size[i])
        
        # Sum DCG
        for j in range(text2_size[i]):
            sum_DCG_Arr[j] += DCG_Arr[j]
        
        # Ideal Ranking, descending order
        text2_match_text1_Arr[::-1].sort(axis = 0)
        #print(text2_match_text1_Arr)
        
        # ICG
        ICG_Arr = np.zeros(text2_size[i], dtype = 'f')
        ICG_Arr = Cumulated_Gain(text2_match_text1_Arr, text2_size[i])
        
        # IDCG
        IDCG_Arr = np.zeros(text2_size[i], dtype = 'f')
        IDCG_Arr = Discounted_Cumulated_Gain(ICG_Arr, text2_size[i])
        
        # Sum IDCG
        for j in range(text2_size[i]):
            sum_IDCG_Arr[j] += IDCG_Arr[j]
    
    try:
        avg_DCG_Arr = sum_DCG_Arr / query_size
    except ZeroDivisionError as devide_by_zero:
        print(devide_by_zero)
    try:
        avg_IDCG_Arr = sum_IDCG_Arr / query_size
    except ZeroDivisionError as devide_by_zero:
        print(devide_by_zero)
    
    NDCG_Arr = np.zeros(max_document_size, dtype = 'f')
    NDCG_Arr = avg_DCG_Arr / avg_IDCG_Arr
    
    #print(NDCG_Arr)
    
    # plot NDCG
    plt.style.use('seaborn-whitegrid')
    plt.plot(NDCG_Arr)
    plt.ylim(0,1)
    my_x_ticks = np.arange(0, max_document_size, 250)
    my_y_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.xlabel('Position(p)')
    plt.ylabel('NDCG')
    plt.title('Normalized_DCG')
    plt.savefig('HW1_Normalized_DCG.png')
    plt.show()
   
#==============================================================================
if __name__ == '__main__':

    # Read AssessmentTrainSet.txt
    try:
        f_assessment = open(r'./AssessmentTrainSet.txt', 'r')
        text_assesssment, query_assesment_size = Read_File(f_assessment)
    except OSError:
         print('Cannot open AssessmentTrainSet.txt')
    
    # Read ResultsTrainSet.txt
    try:
        f_results = open(r'./ResultsTrainSet.txt', 'r')
        text_results, query_results_size = Read_File(f_results)
    except OSError:
        print('Cannot open ResultsTrainSet.txt')
    
    # Task 1, 2, 3
    Interpolated_Recall_Precision_Curve(text_assesssment, query_assesment_size, text_results, query_results_size)
    Mean_Average_Precision(text_assesssment, query_assesment_size, text_results, query_results_size)
    Normalized_DCG(text_assesssment, query_assesment_size, text_results, query_results_size)
