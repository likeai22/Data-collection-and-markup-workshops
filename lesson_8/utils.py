import numpy as np
import pandas as pd
import unidecode

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Константы
TOP_N_CORRELATIONS = 5
CORRELATION_THRESHOLD = 0.7
P_VALUE_THRESHOLD = 0.05


def optimizing_df(df: pd.DataFrame):
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


def reduce_mem_usage(df: pd.DataFrame, use_float16=False):
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
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    use_float16
                    and c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Использование памяти после оптимизации: {:.2f} MB".format(end_mem))
    print("Снизилось на {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df


def show(dataframe: pd.DataFrame):
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
def clean_columns(df: pd.DataFrame, columns):
    for col in columns:
        df[col] = clean_string(df[col])
    return df


def remove_uninformative_columns(
    dataset: pd.DataFrame,
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
    df: pd.DataFrame, column_name, lower_quantile=0.005, upper_quantile=0.995
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
    print(f"Нижний квантиль ({lower_quantile * 100}%): {lower_bound}")
    print(f"Верхний квантиль ({upper_quantile * 100}%): {upper_bound}")
    print(f"Процент удаленных данных: {percent_removed:.2f}%")

    return filtered_df, percent_removed


# stackoverflow код удаляет колонки с корреляцией > threshold
def correlation(dataset: pd.DataFrame, threshold):
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


def srt_box(y, data: pd.DataFrame):
    fig, axes = plt.subplots(12, 3, figsize=(25, 80))
    axes = axes.flatten()

    for i, j in zip(data.select_dtypes(include=["category"]).columns, axes):

        sortd = data.groupby([i])[y].median().sort_values(ascending=False)
        sns.boxplot(x=i, y=y, data=data, palette="plasma", order=sortd.index, ax=j)
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=18))

        plt.tight_layout()


def analyze_categorical_columns(df: pd.DataFrame):
    for column in df.columns:
        if df[column].dtype == "object" or df[column].dtype == "category":
            unique_values = df[column].unique()
            num_unique_values = len(unique_values)

            print(
                f"Колонка: {column}, уникальные значения: {unique_values},"
                f" количество: {num_unique_values}"
            )
            print("=" * 40)


# Constants


def calculate_correlations(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Calculate Spearman and Kendall correlation coefficients and p-values for all numeric columns.

    Args:
    data (pd.DataFrame): Input dataframe
    target (str): Name of the target variable

    Returns:
    pd.DataFrame: Dataframe with correlation coefficients and p-values
    """
    quantitative_vars = data.select_dtypes(include=[np.number])
    results = {}

    for col in quantitative_vars.columns:
        if col != target:
            spearman_corr, spearman_p_value = stats.spearmanr(
                data[target], data[col], nan_policy="omit"
            )
            kendall_corr, kendall_p_value = stats.kendalltau(
                data[target], data[col], nan_policy="omit"
            )
            results[col] = {
                "Spearman_corr": spearman_corr,
                "Spearman_p_value": spearman_p_value,
                "Kendall_corr": kendall_corr,
                "Kendall_p_value": kendall_p_value,
            }

    return pd.DataFrame.from_dict(results, orient="index")


def filter_significant_correlations(
    correlations: pd.DataFrame, method: str = "Spearman"
) -> pd.DataFrame:
    """
    Фильтрация корреляций на основе пороговых значений.

    Аргументы:
    correlations (pd.DataFrame): Входной датафрейм с корреляциями
    method (str): Метод корреляции для использования ('Spearman' или 'Kendall')

    Возвращает:
    pd.DataFrame: Отфильтрованный датафрейм с корреляциями
    """
    corr_column = f"{method}_corr"
    p_value_column = f"{method}_p_value"

    return correlations[
        (correlations[corr_column].abs() > CORRELATION_THRESHOLD)
        & (correlations[p_value_column] <= P_VALUE_THRESHOLD)
    ]


def visualize_correlations(correlations: pd.DataFrame, title: str):
    """
    Создание тепловой карты коэффициентов корреляции.

    Аргументы:
    correlations (pd.DataFrame): Входной датафрейм с корреляциями
    title (str): Заголовок для тепловой карты
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlations[["Spearman_corr", "Kendall_corr"]],
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.title(title)
    plt.show()


def correlation_analysis(data: pd.DataFrame, target: str):
    """
    Основная функция для проведения анализа корреляции.

    Аргументы:
    data (pd.DataFrame): Входной датафрейм
    target (str): Название целевой переменной
    """
    try:
        # Check for missing values
        if data.isnull().sum().sum() > 0:
            print("Warning: Dataset contains missing values. Results may be affected.")

        correlations = calculate_correlations(data, target)
        print(f"Correlation coefficients and p-values with {target}:")
        print(correlations)

        significant_correlations = filter_significant_correlations(correlations)
        print(f"\nSignificant high correlations with {target}:")
        print(significant_correlations)

        visualize_correlations(correlations, f"Correlation Heatmap with {target}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def anova(frame: pd.DataFrame, numerical_cols: pd.DataFrame) -> pd.DataFrame:
    """
    Проведение дисперсионного анализа (ANOVA) для набора числовых признаков.

    Аргументы:
    frame (pd.DataFrame): Входной датафрейм
    numerical_cols (pd.DataFrame): Датафрейм с числовыми колонками

    Возвращает:
    pd.DataFrame: Датафрейм с именами признаков и соответствующими значениями p-value, отсортированный по возрастанию p-value
    """
    anv = pd.DataFrame()
    anv["feature"] = numerical_cols.columns.tolist()
    pvals = []

    for c in numerical_cols.columns.tolist():
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]["SalePrice"].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)

    anv["pval"] = pvals
    return anv.sort_values("pval")


def calculate_and_plot_correlations(data: pd.DataFrame, target: str):
    """
    Рассчитать коэффициенты корреляции с целевой переменной и построить тепловую карту для значительных корреляций.

    Аргументы:
    data (pd.DataFrame): Входной датафрейм
    target (str): Название целевой переменной

    Возвращает:
    None: Функция выводит тепловую карту.
    """
    correlations = data.corrwith(data[target], numeric_only=True).iloc[:-1].to_frame()
    correlations["Abs Corr"] = correlations[0].abs()
    sorted_correlations = correlations.sort_values("Abs Corr", ascending=False)[
        "Abs Corr"
    ]

    # Построение тепловой карты для коэффициентов корреляции >= 0.5
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        sorted_correlations.to_frame()[sorted_correlations >= 0.5],
        cmap="coolwarm",
        annot=True,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    plt.show()


def plot_top_correlations(
    data: pd.DataFrame, numerical_cols: pd.DataFrame, target: str
):
    """
    Построить графики распределений для признаков с наивысшими корреляциями с целевой переменной.

    Аргументы:
    data (pd.DataFrame): Входной датафрейм
    numerical_cols (pd.DataFrame): Датафрейм с числовыми колонками
    target (str): Название целевой переменной

    Возвращает:
    None: Функция выводит графики распределений.
    """
    # Рассчитать корреляции с целевой переменной
    correlations = data.corrwith(data[target], numeric_only=True).iloc[:-1].to_frame()
    correlations["Abs Corr"] = correlations[0].abs()
    top_correlations = correlations.sort_values("Abs Corr", ascending=False).head(
        TOP_N_CORRELATIONS
    )

    # Построить графики для признаков с наивысшими корреляциями
    f = pd.melt(data, value_vars=top_correlations.index)
    g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False)
    g.map(sns.histplot, "value", kde=True)
    plt.show()


def encode_categorical_features(frame: pd.DataFrame, feature: str, target: str):
    """
    Кодировать категориальные признаки на основе среднего значения целевой переменной.

    Аргументы:
    frame (pd.DataFrame): Входной датафрейм
    feature (str): Название категориального признака
    target (str): Название целевой переменной

    Возвращает:
    None: Функция добавляет закодированные признаки в исходный датафрейм.
    """
    ordering = pd.DataFrame()
    ordering["val"] = frame[feature].unique()
    ordering.index = ordering.val
    ordering["spmean"] = frame[[feature, target]].groupby(feature).mean()[target]
    ordering = ordering.sort_values("spmean")
    ordering["ordering"] = range(1, ordering.shape[0] + 1)
    ordering = ordering["ordering"].to_dict()

    frame[feature + "_E"] = frame[feature].map(ordering)


def encode_qualitative_features(
    frame: pd.DataFrame, string_cols: pd.DataFrame, target: str
) -> list:
    """
    Кодировать все категориальные признаки в датафрейме.

    Аргументы:
    frame (pd.DataFrame): Входной датафрейм
    string_cols (pd.DataFrame): Датафрейм с категориальными признаками
    target (str): Название целевой переменной

    Возвращает:
    list: Список названий закодированных признаков.
    """
    qual_encoded = []
    for q in string_cols.columns:
        encode_categorical_features(frame, q, target)
        qual_encoded.append(q + "_E")
    return qual_encoded


def plot_correlation_heatmap(
    data: pd.DataFrame, figsize: tuple = (20, 20), font_scale: float = 1.1
):
    """
    Построение тепловой карты коэффициентов корреляции для числовых признаков в датафрейме.

    Аргументы:
    data (pd.DataFrame): Входной датафрейм
    figsize (tuple): Размер фигуры для графика (по умолчанию (20, 20))
    font_scale (float): Масштаб шрифта для графика (по умолчанию 1.1)

    Возвращает:
    None: Функция выводит тепловую карту.
    """
    sns.set(font_scale=font_scale)
    correlation_matrix = data.corr(numeric_only=True)
    mask = np.triu(
        np.ones_like(correlation_matrix, dtype=bool)
    )  # Верхний треугольник маскируется

    plt.figure(figsize=figsize)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        square=True,
        mask=mask,
        linewidths=1,
        cbar=False,
    )
    plt.title(
        "Тепловая карта коэффициентов корреляции для числовых признаков в датафрейме"
    )
    plt.show()


def analyze_categorical_correlations(
    data: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Анализ взаимосвязей между категориальными переменными в датафрейме.

    Аргументы:
    data (pd.DataFrame): Входной датафрейм
    alpha (float): Уровень значимости для статистических тестов (по умолчанию 0.05)

    Возвращает:
    pd.DataFrame: Датафрейм с результатами тестов на независимость для каждой пары категориальных переменных.
    """
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    results = []

    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i + 1 :]:
            contingency_table = pd.crosstab(data[col1], data[col2])
            chi2, p, _, _ = stats.chi2_contingency(contingency_table)
            cramers_v = np.sqrt(
                chi2 / (data.shape[0] * (min(contingency_table.shape) - 1))
            )

            results.append(
                {
                    "Variable 1": col1,
                    "Variable 2": col2,
                    "Chi2": chi2,
                    "p-value": p,
                    "Cramér's V": cramers_v,
                    "Significant": p < alpha,
                }
            )

    return pd.DataFrame(results).sort_values(by="Cramér's V", ascending=False)


def cramers_v_heatmap(
    data: pd.DataFrame, figsize: tuple = (20, 20), font_scale: float = 1.1
):
    """
    Построение тепловой карты коэффициентов Крамера V для категориальных переменных в датафрейме.

    Аргументы:
    data (pd.DataFrame): Входной датафрейм
    figsize (tuple): Размер фигуры для графика (по умолчанию (20, 20))
    font_scale (float): Масштаб шрифта для графика (по умолчанию 1.1)

    Возвращает:
    None: Функция выводит тепловую карту.
    """
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    n = len(categorical_cols)
    cramers_v_matrix = np.zeros((n, n))

    for i, col1 in enumerate(categorical_cols):
        for j, col2 in enumerate(categorical_cols):
            if i == j:
                cramers_v_matrix[i, j] = 1.0  # Коэффициент для самой переменной равен 1
            else:
                contingency_table = pd.crosstab(data[col1], data[col2])
                chi2, _, _, _ = stats.chi2_contingency(contingency_table)
                cramers_v = np.sqrt(
                    chi2 / (data.shape[0] * (min(contingency_table.shape) - 1))
                )
                cramers_v_matrix[i, j] = cramers_v

    cramers_v_df = pd.DataFrame(
        cramers_v_matrix, index=categorical_cols, columns=categorical_cols
    )

    sns.set(font_scale=font_scale)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cramers_v_df,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        square=True,
        linewidths=1,
        cbar=False,
    )
    plt.title(
        "Тепловая карта коэффициентов Крамера V для категориальных переменных в датафрейме"
    )
    plt.show()


def encode_features(
    data: pd.DataFrame,
    label_encoding_list: list,
    one_hot_encoding_list: list,
    cleaned_string_cols: list,
    cleaned_numeric_cols: list,
) -> pd.DataFrame:
    """
    Кодирует категориальные переменные с использованием Label Encoding и One-Hot Encoding.

    Аргументы:
    data (pd.DataFrame): Входной датафрейм с категориальными и числовыми переменными.
    label_encoding_list (list): Список переменных для кодирования с помощью Label Encoding.
    one_hot_encoding_list (list): Список переменных для кодирования с помощью One-Hot Encoding.
    cleaned_string_cols (pd.Index): Индекс с именами всех категориальных столбцов.
    cleaned_numeric_cols (pd.Index): Индекс с именами всех числовых столбцов.

    Возвращает:
    pd.DataFrame: Датафрейм с закодированными и числовыми переменными.
    """
    onehot_encoder = OneHotEncoder(sparse_output=False, drop="first")
    label_encoder = LabelEncoder()

    encoded_data = pd.DataFrame(
        index=data.index
    )  # Инициализация с сохранением индексов

    for col in cleaned_string_cols.columns.tolist():
        if col in label_encoding_list:
            # Label Encoding для порядковых переменных
            encoded_data[col] = label_encoder.fit_transform(data[col])
        elif col in one_hot_encoding_list:
            # One-Hot Encoding для номинальных переменных
            encoded_cols = onehot_encoder.fit_transform(data[[col]])
            col_names = [f"{col}_{cat}" for cat in onehot_encoder.categories_[0][1:]]
            encoded_df = pd.DataFrame(encoded_cols, columns=col_names, index=data.index)
            encoded_data = pd.concat([encoded_data, encoded_df], axis=1)
        else:
            # Если переменная не нуждается в кодировании, добавляем её как есть
            encoded_data[col] = data[col]

    # Добавление числовых переменных обратно в набор данных
    final_data = pd.concat(
        [encoded_data, data[cleaned_numeric_cols.columns.tolist()]], axis=1
    )

    return final_data


def plot_distribution_analysis(df: pd.DataFrame, feature: str, title: str) -> None:
    """
    Создает комбинированный график распределения, включающий гистограмму, график вероятностей (QQ Plot) и коробчатую диаграмму (Box Plot).

    Аргументы:
    df (pd.DataFrame): Входной датафрейм с данными.
    feature (str): Название столбца для анализа.
    title (str): Заголовок для всего графика.

    Возвращает:
    None: Функция выводит комбинированный график.
    """
    # Инициализация фигуры и сетки для размещения подграфиков
    fig = plt.figure(constrained_layout=True, figsize=(14, 8))
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Гистограмма и KDE
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title("Гистограмма и KDE", fontsize=16)
    sns.histplot(
        df[feature].dropna(), kde=True, ax=ax1, color="#e74c3c", stat="density"
    )
    sns.kdeplot(df[feature].dropna(), ax=ax1, color="blue", lw=2)
    ax1.legend(labels=["KDE", "Фактические данные"])

    # QQ Plot
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title("График вероятностей (QQ Plot)", fontsize=16)
    stats.probplot(df[feature].dropna(), dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor("#e74c3c")
    ax2.get_lines()[0].set_markersize(8.0)

    # Box Plot
    ax3 = fig.add_subplot(grid[:, 2])
    ax3.set_title("Коробчатая диаграмма (Box Plot)", fontsize=16)
    sns.boxplot(y=df[feature], ax=ax3, color="#e74c3c")
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=10))

    # Общий заголовок для всей фигуры
    plt.suptitle(title, fontsize=20)
    plt.show()
