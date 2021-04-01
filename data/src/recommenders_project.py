import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix, coo_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка (построение own_recommender)
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from lightgbm import LGBMClassifier
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, user_features, item_features, items_to_filter=[999999], weighting=True):
        
        self.items_to_filter = items_to_filter
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[~self.overall_top_purchases['item_id'].isin(self.items_to_filter)]  # ~self.top_purchases отрицание
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix, self.sparse_user_item = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        # LightFM не будет взвешен при такой конструкции
        self.user_feat_lightfm_fixed, self.item_feat_lightfm_fixed = self._prepare_user_item_feat_lightfm(self.user_item_matrix, user_features, item_features)
        self.user_item_matrix_lightfm = self.user_item_matrix.copy()

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

    @staticmethod
    def _prepare_matrix(data):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        # переведем в формат sparse matrix (для LightFM)
        sparse_user_item = csr_matrix(user_item_matrix).tocsr()

        return user_item_matrix, sparse_user_item

    @staticmethod
    def _prepare_dicts(user_item_matrix):
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
    def _prepare_user_item_feat_lightfm(user_item_matrix, user_features, item_features):
        """Готовит под нужный формат фичи для LightFM"""

        user_feat = pd.DataFrame(user_item_matrix.index)
        user_feat = user_feat.merge(user_features, on='user_id', how='left').drop(columns=['homeowner_desc'])
        user_feat.set_index('user_id', inplace=True)

        item_feat = pd.DataFrame(user_item_matrix.columns)
        item_feat = item_feat.merge(item_features, on='item_id', how='left').drop(columns=['sub_commodity_desc', 'curr_size_of_product'])
        item_feat.set_index('item_id', inplace=True)

        user_feat_lightfm_fixed = pd.get_dummies(user_feat, columns=user_feat.columns.tolist())
        item_feat_lightfm_fixed = pd.get_dummies(item_feat, columns=item_feat.columns.tolist())

        return user_feat_lightfm_fixed, item_feat_lightfm_fixed

    def fit_own_recommender(self):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        self.own_recommender = ItemItemRecommender(K=1, num_threads=4)
        self.own_recommender.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        return self.own_recommender

    def fit_als(self, n_factors=20, regularization=0.001, iterations=15, num_threads=4, show_progress=True, use_gpu=True, random_state=42):
        """Обучает ALS"""

        self.model_als = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        use_gpu=use_gpu,
                                        num_threads=num_threads, random_state=random_state)
        self.model_als.fit(csr_matrix(self.user_item_matrix).T.tocsr(), show_progress=show_progress)

        return self.model_als

    def fit_lightfm(self, no_components=16, loss='warp', learning_rate=0.05, item_alpha=0.2, user_alpha=0.05, random_state=42, epochs=15):
        """Обучает LightFM"""

        self.model_lightfm = LightFM(no_components=no_components,
                        loss=loss,  # или 'warp' - ниже в уроке описана разница
                        learning_rate=learning_rate,
                        item_alpha=item_alpha, user_alpha=user_alpha,
                        random_state=random_state)

        self.model_lightfm.fit((self.sparse_user_item > 0) * 1,  # user-item matrix из 0 и 1
                        sample_weight=coo_matrix(self.user_item_matrix_lightfm),  # матрица весов С
                        user_features=csr_matrix(self.user_feat_lightfm_fixed.values).tocsr(),
                        item_features=csr_matrix(self.item_feat_lightfm_fixed.values).tocsr(),
                        epochs=epochs,
                        num_threads=8)

        return self.model_lightfm

    def precision_at_k_lightfm(self, model_lightfm, sparse_user_item, k=5):
        """Precision встроенный в LightFM"""
        
        self.precision_res = precision_at_k(model_lightfm, sparse_user_item, 
                                 user_features=csr_matrix(self.user_feat_lightfm_fixed.values).tocsr(),
                                 item_features=csr_matrix(self.item_feat_lightfm_fixed.values).tocsr(),
                                 k=k)

        return self.precision_res

    def recall_at_k_lightfm(self, model_lightfm, sparse_user_item, k=5):
        """Recall встроенный в LightFM"""

        self.recall_res = recall_at_k(model_lightfm, sparse_user_item, 
                                 user_features=csr_matrix(self.user_feat_lightfm_fixed.values).tocsr(),
                                 item_features=csr_matrix(self.item_feat_lightfm_fixed.values).tocsr(),
                                 k=k)

        return self.recall_res

    def _update_dict(self, user_id):
        """Если появился новый user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""

        recs = self.model_als.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""
        
        self._update_dict(user_id=user)
        
        recs = model.recommend(userid=self.userid_to_id[user],
                                        user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                        N=N,
                                        filter_already_liked_items=False,
                                        filter_items=[self.itemid_to_id[item] for item in self.items_to_filter if item in self.itemid_to_id.keys()],  # [self.itemid_to_id[item] for item in items_to_filter if item in self.itemid_to_id.keys()]
                                        recalculate_user=True)
        
        res = [self.id_to_itemid[rec[0]] for rec in recs]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model_als, N=N)

    def get_lightfm_recommendations(self, user, N=5):
        """Рекомендации для библиотеки LightFM"""

        self._update_dict(user_id=user)
        test_item_ids = np.arange(len(self.itemid_to_id))

        scores = self.model_lightfm.predict(user_ids=int(self.userid_to_id[user]),  # На <class 'numpy.int64'> ругается, поэтому в int перевожу
                                    item_ids=test_item_ids,
                                    user_features=csr_matrix(self.user_feat_lightfm_fixed.values).tocsr(),
                                    item_features=csr_matrix(self.item_feat_lightfm_fixed.values).tocsr(),
                                    num_threads=8)
        top_items = np.argsort(-scores)
        
        res = [self.id_to_itemid[item] for item in top_items][:N]  # Конвертируем id обратно, делаем срез на нужное число, т.к. предсказания были для всех item    
        res = self._extend_with_top_popular(res, N=N)  # Теоретически для этой модели не нужно, т.к. предсказания сразу для всех делает?

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров.
        Не фильтрует item_id!"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами.
        Не фильтрует item_id!"""

        res = []
        
        # Находим топ-N похожих пользователей
        similar_users = self.model_als.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]   # удалим юзера из запроса

        for user in similar_users:
            userid = self.id_to_userid[user] # own recommender works with user_ids
            res.extend(self.get_own_recommendations(userid, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res