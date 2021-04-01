import pandas as pd

def prefilter_items(data_train, item_features):
    # Оставим только 5000 самых популярных товаров
    popularity = data_train.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_5000 = popularity.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()
    #добавим, чтобы не потерять юзеров
    data_train.loc[~data_train['item_id'].isin(top_5000), 'item_id'] = 999999 
     
    # Уберем самые популярные
    top_corrected = top_5000[10:]
    
    # Уберем самые непопулярные
    top_corrected = top_corrected[:-10]
    data_train.loc[~data_train['item_id'].isin(top_corrected), 'item_id'] = 999999 
    
    # Уберем товары, которые не продавались за последние 12 месяцев (это примерно 52 недели)
    data_train.loc[data_train['week_no'] < data_train['week_no'].max() - 52, 'item_id'] = 999999  # Оставляю продажи после 39 недели
    
    # Уберем не интересные для рекоммендаций категории (department)
    data_train_dep = pd.merge(data_train, item_features[['item_id', 'department']], how='left', on="item_id")
    bad_departments = ['KIOSK-GAS']
    for dept in bad_departments:
        data_train_dep.loc[data_train_dep['department'] == dept, 'item_id'] = 999999
    data_train = data_train_dep.drop(columns=['department'])

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data_train.loc[data_train['sales_value'] < 6, 'item_id'] = 999999
    
    # Уберем слишком дорогие товары
    data_train.loc[data_train['sales_value'] > 500, 'item_id'] = 999999

    return data_train


def postfilter_items():
    pass


def get_rec(model, x):

    recs = model.similar_items(itemid_to_id[x], N=3)
    top_rec = recs[1][0] if id_to_itemid[recs[1][0]] != 999999 else recs[2][0]

    return id_to_itemid[top_rec]


def get_recommendations(user, model, sparse_user_item, N=5):
    """Рекомендуем топ-N товаров"""
    res = [id_to_itemid[rec[0]] for rec in model.recommend(userid=userid_to_id[user], 
                                    user_items=sparse_user_item,   # на вход user-item matrix
                                    N=N, 
                                    filter_already_liked_items=False, 
                                    filter_items=[itemid_to_id[999999]],  # !!! Используем фильтр
                                    recalculate_user=True)]
    return res


def get_similar_items_recommendation(user, model, N=5):
    """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

    popularity_n_items = popularity.head(N)  # Оставляем первые N самых "популярных" товаров
    res = popularity_n_items[popularity_n_items['user_id'] == user]['item_id'].apply(lambda x: get_rec(model, x))  # Результат для конкретного юзера
   
    return res.tolist()


def get_similar_users_recommendation(user, model, N=5):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
    res = []
    for matrix_userid in model.similar_users(userid_to_id[user], N=N+1):
        if matrix_userid[0] != userid_to_id[user]:
            res.append(get_recommendations(user=id_to_userid[matrix_userid[0]], model=own, sparse_user_item=csr_matrix(user_item_matrix).T.tocsr(), N=1)[0])
    
    return res
