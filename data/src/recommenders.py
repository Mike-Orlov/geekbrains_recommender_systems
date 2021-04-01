import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, item_features, weighting=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        data = self.prefilter_items(data, item_features)

        # predefined code
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

        # вынес сюда группировку, чтобы не делать её каждый раз при вызове метода get_similar_items_recommendation
        self.popularity = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.popularity.sort_values('quantity', ascending=False, inplace=True)  # "Популярность" по кол-ву покупок

        item_filter = 999999
        if item_filter:
            self.popularity = self.popularity[self.popularity['item_id'] != item_filter]  # item_filter = 999999, dummy item
        self.popularity = self.popularity.groupby('user_id')  # Заранее сгруппируем по юзерам

    @staticmethod
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

    @staticmethod
    def postfilter_items():
        pass

    @staticmethod
    def prepare_matrix(data):
        
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id', columns='item_id', 
                                  values='quantity', # Можно пробовать другие варианты
                                  aggfunc='count', 
                                  fill_value=0
                                 )

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,
                                             use_gpu=False,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    @staticmethod
    def get_rec(model, x):
        '''Рекомендует 1 наиболее похожий item'''

        recs = model.similar_items(itemid_to_id[x], N=3)
        top_rec = recs[1][0] if id_to_itemid[recs[1][0]] != 999999 else recs[2][0]  # Фильтрую dummy item 999999, для этого выше N=3, чтобы был альтернативный вариант

        return id_to_itemid[top_rec]

    @staticmethod
    def get_recommendations(user, model, sparse_user_item, N=5):
        """Рекомендуем топ-N товаров"""
        res = [id_to_itemid[rec[0]] for rec in 
                        model.recommend(userid=userid_to_id[user], 
                                        user_items=sparse_user_item,   # на вход user-item matrix
                                        N=N, 
                                        filter_already_liked_items=False, 
                                        filter_items=[itemid_to_id[999999]],  # !!! Используем фильтр
                                        recalculate_user=True)]
        return res

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        self.popularity_n_items = self.popularity.head(N)  # Оставляем первые N самых "популярных" товаров

        try:
            res = self.popularity_n_items[self.popularity_n_items['user_id'] == user]['item_id'].apply(lambda x: get_rec(self.model, x))  # Результат для конкретного юзера

            res = res.tolist()

            #  Пока закостылю, надо подумать, при user=10 только 1 рекомендация
            if len(res) < N:
                for i in range(N - len(res)):
                    res.append(999999)
        except:
            res = [999999 for i in range(N)]  # Потом поправлю, ошибка для user=2325 KeyError, запусти userid_to_id[2325]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []
        try:
            for matrix_userid in model.similar_users(userid_to_id[user], N=N+1):
                if matrix_userid[0] != userid_to_id[user]:
                    if len(get_recommendations(user=id_to_userid[matrix_userid[0]], model=own, sparse_user_item=csr_matrix(user_item_matrix).T.tocsr(), N=1)) != 0:
                        res.append(get_recommendations(user=id_to_userid[matrix_userid[0]], model=own, sparse_user_item=csr_matrix(user_item_matrix).T.tocsr(), N=1)[0])
                    else:
                        res.append(999999)
        except:
            res = [999999 for i in range(N)]  # Потом поправлю, ошибка для user=2325 KeyError, запусти userid_to_id[2325]

        res = res[:N] if len(res) > N else res  # См. 102 - 6 рекомендаций

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res