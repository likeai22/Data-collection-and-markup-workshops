import requests
from lxml import html
import csv
from fake_useragent import UserAgent
import flet as ft

# Константы
URL: str = "https://www.trustbit.tech/en/llm-leaderboard-juli-2024"
CSV_FILENAME: str = "llm_leaderboard_data.csv"


def fetch_html(url: str) -> bytes | None:
    """
    Получает HTML-контент с указанного URL.

    Аргументы:
        url: URL веб-сайта.

    Возвращает:
        HTML-контент страницы в виде байтов или None, если произошла ошибка.
    """
    ua = UserAgent()
    headers = {
        "User-Agent": ua.random
    }  # Используем случайный User-Agent для обхода блокировок
    try:
        response = requests.get(url, headers=headers)  # Отправляем GET-запрос
        response.raise_for_status()  # Вызывает исключение, если статус код не 200
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None


def parse_html(html_content: bytes | None) -> html.HtmlElement | None:
    """
    Парсит HTML-контент с помощью библиотеки lxml.

    Аргументы:
        html_content: HTML-контент для парсинга.

    Возвращает:
        Разобранное дерево HTML или None, если произошла ошибка.
    """
    if html_content is None:
        return None
    return html.fromstring(html_content)  # Преобразуем байты в дерево HTML


def extract_table_data(
        tree: html.HtmlElement | None,
) -> tuple[list[str] | None, list[list[str]] | None]:
    """
    Извлекает данные таблицы с помощью XPath выражений.

    Аргументы:
        tree: Разобранное дерево HTML.

    Возвращает:
        Кортеж, содержащий заголовки таблицы и данные.
        Возвращает (None, None), если произошла ошибка.
    """
    if tree is None:
        return None, None
    table_rows = tree.xpath(
        '//table[@class="custom-model-table"]//tr'
    )  # Находим все строки таблицы
    if not table_rows:
        print("Таблица не найдена на странице. Проверьте выражение XPath.")
        return None, None
    headers = [
        header.text.strip() for header in table_rows[0].xpath(".//th")
    ]  # Извлекаем заголовки
    data = [
        [cell.text.strip() for cell in row.xpath(".//td")]  # Извлекаем данные из ячеек
        for row in table_rows[1:]  # Пропускаем первую строку, так как она содержит заголовки
        if len(row) > 0  # Проверяем, что строка содержит ячейки
    ]
    return headers, data


def save_to_csv(
        filename: str, headers: list[str] | None, data: list[list[str]] | None
) -> None:
    """
    Сохраняет извлеченные данные в CSV-файл.

    Аргументы:
        filename: Имя файла CSV.
        headers: Заголовки таблицы.
        data: Данные таблицы.
    """
    if headers is None or data is None:
        print(
            "Ошибка: Невозможно сохранить данные в CSV. Отсутствуют заголовки или данные."
        )
        return
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)  # Записываем заголовки
        csvwriter.writerows(data)  # Записываем строки данных


def main(page: ft.Page):
    page.title = "Web Scraper"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    url_input = ft.TextField(label="Введите URL", value=URL, width=500)
    status_label = ft.Text()

    def start_scraping(e):
        try:
            url = url_input.value
            status_label.value = "Получение данных..."
            page.update()

            html_content = fetch_html(url)
            tree = parse_html(html_content)
            headers, data = extract_table_data(tree)
            save_to_csv(CSV_FILENAME, headers, data)
            status_label.value = f"Данные успешно сохранены в {CSV_FILENAME}"
            page.update()
        except requests.exceptions.HTTPError as http_err:
            status_label.value = f"Произошла HTTP ошибка: {http_err}"
            page.update()
        except requests.exceptions.RequestException as req_err:
            status_label.value = f"Произошла ошибка запроса: {req_err}"
            page.update()
        except Exception as e:
            status_label.value = f"Произошла непредвиденная ошибка: {e}"
            page.update()

    page.add(
        ft.Row(
            [url_input, ft.ElevatedButton("Начать парсинг", on_click=start_scraping)]
        ),
        status_label,
    )


if __name__ == "__main__":
    ft.app(target=main)
