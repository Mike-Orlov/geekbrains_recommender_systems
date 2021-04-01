import numpy as np


def precision(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(recommended_list, bought_list)
    precision = flags.sum() / len(recommended_list)

    return precision


def money_precision_at_k(recommended_list, bought_list, prices_bought, k=5):

    # your_code
    # Лучше считать через скалярное произведение, а не цикл
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    prices_bought_list = np.array(prices_bought)

    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    prices_bought_list = prices_bought_list  # Тут нет [:k] !!

    flags = np.isin(recommended_list, bought_list)

    precision = np.sum(flags*prices_bought_list) / np.sum(prices_bought_list)

    return precision


def hit_rate(recommended_list, bought_list):

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    hit_rate = (flags.sum() > 0).astype(int)

    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    flags = np.isin(bought_list, recommended_list)
    
    hit_rate = (flags.sum() > 0).astype(int)
    
    return hit_rate


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    recall = flags.sum() / len(bought_list)
    
    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)
    
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    # your_code

    return recall


def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(1, k+1):
        if flags[i] is True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k
            
    result = sum_ / sum(flags)
    
    return result


def reciprocal_rank(recommended_list, bought_list, k=5):
    
    recommended_list = recommended_list[:k]
    ranks = 0
    for i, item_rec in enumerate(recommended_list):
        for item_bought in bought_list:
            if item_rec == item_bought:
                ranks += 1 / (i+1)
                
    return ranks / len(recommended_list)


def normalized_discounted_cumulative_gain(recommended_list, bought_list, k=5):
    
    dcg, idcg = 0, 0
    dcg_list = []
    idcg_list = []
    recommended_list = recommended_list[:k]
    
    flags = np.isin(recommended_list, bought_list)
    
    for i in range(len(flags)):
        if i == 0:
            dcg_list.append(flags[i]/1)
            idcg_list.append(1/1)
        else:
            dcg_list.append(flags[i]/np.log2(i+1))
            idcg_list.append(1/np.log2(i+1))
    
    return np.mean(np.array(dcg_list)) / np.mean(np.array(idcg_list))
