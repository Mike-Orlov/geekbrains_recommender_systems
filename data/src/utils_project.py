import pandas as pd
import numpy as np

def prefilter_items(data, take_n_popular=5000, item_features=None, bad_departments=None):
    """
    Бизнес-ограничения:
    - Нельзя рекомендовать top 3 самых популярных товаров
    - Нельзя рекомендовать товары, которые стоят < 1$
    - Нельзя рекомендовать товары, которые не продавались последние 12 месяцев
    - Нельзя рекомендовать товары, с общим числом продаж < 50

    Параметры:
    - item_features: если передать фичи, то убрет категории, где мало товаров, по умолчанию менее 150
    - bad_departments: принимает лист с категориями, которые здесь можно явно исключить
    """

    # Уберем товары с общим числом продаж < 50
    items_by_quantity = data.groupby('item_id')['quantity'].sum().reset_index()
    items_by_quantity = items_by_quantity.loc[items_by_quantity['quantity'] < 50, 'item_id'].tolist()
    # data = data[~data['item_id'].isin(items_by_quantity)]

    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['user_id'] = popularity['user_id'] / data['user_id'].nunique()  # Какая доля юзеров из общего числа покупала этот товар
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    # top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    # data = data[~data['item_id'].isin(top_popular)]
    top_3_popular = popularity.sort_values('share_unique_users', ascending=False).head(3).item_id.tolist()
    #data = data[~data['item_id'].isin(top_3_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    # top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    # data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев (это примерно 52 недели)
    old_items = data.loc[data['week_no'] < data['week_no'].max() - 52, 'item_id'].tolist()
    # data = data[~data['item_id'].isin(old_items)]

    # Уберем не интересные для рекоммендаций категории (department)
    # if item_features is not None:
    #     # Обработаем категории с маленьким числом товаров
    #     department_size = pd.DataFrame(item_features.\
    #                                     groupby('department')['item_id'].nunique().\
    #                                     sort_values(ascending=False)).reset_index()

    #     department_size.columns = ['department', 'n_items']
    #     rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
    #     items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()
    #     data = data[~data['item_id'].isin(items_in_rare_departments)]

    #     # Можно также явно указать категории, которые не интересны
    #     if bad_departments is not None:           
    #         items_in_bad_departments = []
    #         for dept in bad_departments:  #bad_departments = ['KIOSK-GAS', 'PASTRY'] - передать как параметры в функцию
    #             items_in_bad_departments.extend(item_features.loc[item_features['department'] == dept, 'item_id'].tolist())  # Именно extend, чтобы добавить не списки, а их значения

    #         data = data[~data['item_id'].isin(items_in_bad_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    cheap_items = data.loc[data['price'] < 1, 'item_id'].unique().tolist()
    #data = data[data['price'] >= 1]  # Оставляем с ценой не менее 1$

    # Уберем слишком дорогие товары
    # data = data[data['price'] < 50]

    # Возьмем топ по популярности (по умолчанию take_n_popular=5000)
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top_n = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    
    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top_n), 'item_id'] = 999999

    # Собираю лист с товарами, которые нельзя рекоммендовать
    items_to_filter = list(set(items_by_quantity + top_3_popular + old_items + cheap_items + [999999]))  # Drop dublicates

    return data, items_to_filter


def postfilter_items(user_id, recommednations):
    """Бизнес-ограничения:
    - Каждому пользователю нужно порекомендовать 5 товаров, один из них должен быть обязательно товаром, который данный пользователь никогда не покупал
    """
    pass