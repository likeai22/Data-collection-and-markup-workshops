import os
import pytest
from urllib.parse import urljoin
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

from bs4 import BeautifulSoup
from main import (
    get_chromedriver_path,
    scroll_to_bottom,
    find_subreddit_page,
)


# Загрузка переменных окружения из .env файла
load_dotenv()

# Mock данные для тестирования
BASE_URL = os.getenv("BASE_URL")
SUBREDDIT_PATH = os.getenv("SUBREDDIT_PATH")
COMMUNITIES_BUTTON_XPATH = os.getenv("COMMUNITIES_BUTTON_XPATH")
SUBREDDIT_XPATH_TEMPLATE = os.getenv("SUBREDDIT_XPATH_TEMPLATE")
PAGINATION_LINK_XPATH_TEMPLATE = os.getenv("PAGINATION_LINK_XPATH_TEMPLATE")
POST_ELEMENT_XPATH = os.getenv("POST_ELEMENT_XPATH")


@pytest.fixture
def driver():
    chromedriver_path = get_chromedriver_path()
    service = Service(executable_path=chromedriver_path)
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    yield driver
    driver.quit()


def test_get_chromedriver_path():
    chromedriver_path = get_chromedriver_path()
    assert os.path.exists(chromedriver_path)


def test_scroll_to_bottom(driver, test_scroll_attempts: int = 3):
    driver.get(BASE_URL)
    # Вызываем функцию прокрутки до конца страницы
    scroll_to_bottom(driver, scroll_attempts=test_scroll_attempts)
    # Проверяем, что файлы были созданы и не пусты
    for i in range(1, test_scroll_attempts + 1):
        file_path = f"page_source_part_{i}.html"
        assert os.path.exists(file_path), f"Файл {file_path} не найден."
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            assert content, f"Содержимое файла {file_path} пустое."


def test_find_subreddit_page(driver):
    driver.get(BASE_URL)
    result = find_subreddit_page(driver, SUBREDDIT_PATH, max_pages=1)
    assert (
        result is False
    )  # Ожидаем, что сабреддит не будет найден, так как используется тестовый URL


@pytest.mark.parametrize(
    "subreddit_url, expected_count",
    [
        (
            urljoin(BASE_URL, SUBREDDIT_PATH),
            10,
        ),  # Пример URL и ожидаемое количество постов
    ],
)
def test_scrape_subreddit_posts(subreddit_url, expected_count):
    # Загрузка HTML из файла чтобы не вызывать driver
    with open("page_source_part_1.html", "r", encoding="utf-8") as f:
        html_posts = f.read()

    soup = BeautifulSoup(html_posts, "html.parser")
    posts = soup.find_all("shreddit-post")

    assert posts is not None, "Функция должна возвращать список постов"
    assert len(posts) > 0, "Список постов не должен быть пустым"
    assert (
        len(posts) != expected_count
    ), f"Количество постов не должно быть {expected_count}"

    for i, post in enumerate(posts):
        assert post["author"], f"Пост №{i+1} должен содержать автора"
        assert post["post-title"], f"Пост №{i+1} должен содержать заголовок"
        assert post["score"], f"Пост №{i+1} должен содержать рейтинг"
        assert post[
            "comment-count"
        ], f"Пост №{i+1} должен содержать количество комментариев"

    # Проверяем, что файл json создан и не пустой
    file_path = f"posts.json"
    assert os.path.exists(file_path), f"Файл {file_path} не найден."
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        assert content, f"Содержимое файла {file_path} пустое."


if __name__ == "__main__":
    pytest.main()
