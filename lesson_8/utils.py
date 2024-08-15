import numpy as np
import pandas as pd
import unidecode

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


def optimizing_df(df):
    for col in df.columns:
        if df[col].dtypes.kind == "i" or df[col].dtypes.kind == "u":
            if df[col].min() >= 0:
                df[col] = pd.to_numeric(df[col], downcast="unsigned")
            else:
                df[col] = pd.to_numeric(df[col], downcast="integer")

        elif df[col].dtypes.kind == "f" or df[col].dtypes.kind == "c":
            df[col] = pd.to_numeric(df[col], downcast="float")

        elif df[col].dtypes.kind == "O":
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype("category")

    return df


def reduce_mem_usage(df, use_float16=False):
    """Перебрать все столбцы дата-фрейма и изменить тип данных,
    чтобы уменьшить использование памяти.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Использование памяти дата-фрейма составляет {:.2f} MB".format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print("Использование памяти после оптимизации: {:.2f} MB".format(end_mem))
    print("Снизилось на {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df


def show(dataframe):
    return dataframe.loc[::111, :]


# Функция для очистки строк
def clean_string(series):
    # Преобразование всех значений в строки
    series = series.astype(str)
    # Удаление незначащих цифр из названий
    series = series.str.lstrip("1.")
    # Удаление акцентов и спецсимволов
    series = series.apply(unidecode.unidecode)
    # Приведение строки к нижнему регистру, удаление пробелов по краям и новых строк
    series = series.str.strip().str.lower().replace("\n", "")
    # Приведение аббревиатур к общему виду
    series = series.str.replace("football club", "fc")
    series = series.str.replace("sporting club", "sc")
    # Удаление пунктуации
    # series = series.str.replace(r'[^\w\s]', '', regex=True)
    # Удаление дублирующихся пробелов
    series = series.str.replace(r"\s+", " ", regex=True)
    return series


# Функция для применения очистки к указанным столбцам
def clean_columns(df, columns):
    for col in columns:
        df[col] = clean_string(df[col])
    return df


def remove_uninformative_columns(
    dataset,
    uniqueness_threshold=0.9,
    missing_threshold=0.5,
    low_variance_threshold=0.01,
):
    """
    Удаляет неинформативные столбцы на основе различных критериев.

    :param dataset: DataFrame с данными
    :param uniqueness_threshold: Порог уникальности (по умолчанию 0.9)
    :param missing_threshold: Порог пропущенных значений (по умолчанию 0.5)
    :param low_variance_threshold: Порог низкой вариативности (по умолчанию 0.01)
    :return: DataFrame с удалёнными столбцами и список удалённых столбцов
    """
    col_to_remove = set()
    # col_to_remove = []

    # Удаление столбцов с высокой уникальностью
    for col in dataset.select_dtypes(exclude=[np.number]).columns:
        unique_ratio = dataset[col].nunique() / len(dataset)
        if unique_ratio > uniqueness_threshold:
            col_to_remove.add(col)

    # Удаление столбцов с высоким процентом пропущенных значений
    for col in dataset.columns:
        missing_ratio = dataset[col].isnull().mean()
        if missing_ratio > missing_threshold:
            col_to_remove.add(col)

    # Удаление столбцов с низкой вариативностью
    for col in dataset.select_dtypes(exclude=[np.number]).columns:
        most_common = dataset[col].value_counts(normalize=True, dropna=False).values[0]
        if most_common > 1 - low_variance_threshold:
            col_to_remove.add(col)

    # Удаление столбцов с низкой вариативностью
    # for col in dataset.columns:
    #     if dataset[col].dtype == np.number:
    #         most_common = dataset[col].value_counts(normalize=True, dropna=False).values[0]
    #         if most_common > 1 - low_variance_threshold:
    #             col_to_remove.append(col)
    #     else:
    #         most_common = dataset[col].value_counts(normalize=True, dropna=False).values[0]
    #         if most_common > 1 - low_variance_threshold:
    #             col_to_remove.append(col)

    # Удаление неинформативных столбцов из DataFrame
    dataset_cleaned = dataset.drop(columns=col_to_remove)

    return dataset_cleaned, col_to_remove


def remove_extreme_outliers(
    df, column_name, lower_quantile=0.005, upper_quantile=0.995
):
    """
    Функция для удаления экстремальных выбросов из указанного столбца
     датафрейма на основе квантильных значений.

    :param df: DataFrame, из которого нужно удалить выбросы.
    :param column_name: Название столбца, из которого нужно удалить выбросы.
    :param lower_quantile: Нижний квантильный порог для удаления (по умолчанию 0.005).
    :param upper_quantile: Верхний квантильный порог для удаления (по умолчанию 0.995).
    :return: Новый DataFrame без экстремальных выбросов и процент удаленных данных.
    """
    # Вычисление нижнего и верхнего квантилей
    lower_bound = df[column_name].quantile(lower_quantile)
    upper_bound = df[column_name].quantile(upper_quantile)

    # Фильтрация датафрейма на основе квантильных значений
    filtered_df = df[
        (df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)
    ]

    # Вычисление процента удаленных данных
    percent_removed = (df.shape[0] - filtered_df.shape[0]) * 100 / df.shape[0]

    # Вывод результатов
    print(f"Нижний квантиль ({lower_quantile*100}%): {lower_bound}")
    print(f"Верхний квантиль ({upper_quantile*100}%): {upper_bound}")
    print(f"Процент удаленных данных: {percent_removed:.2f}%")

    return filtered_df, percent_removed


# stackoverflow код удаляет колонки с корреляцией > threshold
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (
                corr_matrix.columns[j] not in col_corr
            ):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    # del dataset[colname]  # deleting the column from the dataset
                    dataset = dataset.drop(columns=colname)
                    return dataset
    return dataset


def srt_box(y, data):
    fig, axes = plt.subplots(12, 3, figsize=(25, 80))
    axes = axes.flatten()

    for i, j in zip(data.select_dtypes(include=['category']).columns, axes):

        sortd = data.groupby([i])[y].median().sort_values(ascending=False)
        sns.boxplot(x=i,
                    y=y,
                    data=data,
                    palette='plasma',
                    order=sortd.index,
                    ax=j)
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=18))

        plt.tight_layout()
