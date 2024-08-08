import glob
import json
import os

from lxml import html
from time import sleep
from typing import List, Dict
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Загрузка переменных окружения из .env файла
load_dotenv()

BASE_URL = os.getenv("BASE_URL")
SUBREDDIT_PATH = os.getenv("SUBREDDIT_PATH")
COMMUNITIES_BUTTON_XPATH = os.getenv("COMMUNITIES_BUTTON_XPATH")
SUBREDDIT_XPATH_TEMPLATE = os.getenv("SUBREDDIT_XPATH_TEMPLATE")
PAGINATION_LINK_XPATH_TEMPLATE = os.getenv("PAGINATION_LINK_XPATH_TEMPLATE")
POST_ELEMENT_XPATH = os.getenv("POST_ELEMENT_XPATH")


def get_chromedriver_path() -> str:
    """Возвращает путь к chromedriver."""
    chromedriver_path = "./chromedriver.exe"  # Или "./chromedriver" для MacOS
    if os.path.exists(chromedriver_path):
        return chromedriver_path
    else:
        raise FileNotFoundError(
            "chromedriver не найден. Убедитесь, что он находится в той же директории, что и скрипт."
        )


def scroll_to_bottom(
    driver: webdriver, scroll_attempts: int = 6, scroll_delay: int = 2
) -> None:
    """Прокручивает страницу до конца."""
    for scroll_counter in range(scroll_attempts):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(scroll_delay)
        # Сохранение результатов скраппинга после каждого скролла
        html = driver.page_source
        with open(
            f"page_source_part_{scroll_counter + 1}.html", "w", encoding="utf-8"
        ) as f:
            f.write(html)


def find_subreddit_page(
    driver: webdriver, subreddit_path: str = SUBREDDIT_PATH, max_pages: int = 3
) -> bool:
    """Ищет страницу сабреддита и переходит на нее."""
    for page_num in range(2, max_pages + 1):
        try:
            subreddit_link = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, SUBREDDIT_XPATH_TEMPLATE.format(subreddit_path))
                )
            )
            subreddit_link.click()
            print(f"Сабреддит {subreddit_path} найден!")
            return True
        except Exception:
            try:
                pagination_link = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, PAGINATION_LINK_XPATH_TEMPLATE.format(page_num))
                    )
                )
                pagination_link.click()
                print(f"Переход на страницу {page_num}")
                sleep(2)  # Подождите, пока страница загрузится
            except Exception as e:
                print(f"Страница {page_num} не найдена: {e}")
                break  # Выход из цикла, если страница не найдена
    print(
        f"Сабреддит {subreddit_path} не найден на первых {max_pages} страницах комьюнити."
    )
    return False


def scrape_subreddit_posts(
    driver: webdriver, subreddit_url: str
) -> List[Dict[str, str]] | None:
    """Собирает информацию о постах на странице сабреддита."""
    driver.get(subreddit_url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "shreddit-post"))
    )
    scroll_to_bottom(driver)

    # Список всех сохраненных частей HTML
    html_parts = glob.glob("page_source_part_*.html")
    unique_elements = set()

    for html_part in html_parts:
        # Парсим HTML-документ в объект lxml
        tree = html.parse(html_part)

        # Применяем XPath к документу
        table_rows = tree.xpath(POST_ELEMENT_XPATH)

        # Обрабатываем результаты XPath-запроса (например, выводим их)
        for row in table_rows:
            unique_elements.add(str(html.tostring(row)))

    # Конкатенируем все элементы в один HTML документ
    full_page_source = "".join(list(unique_elements))

    # необходимо по заданию, для использования BeautifulSoup
    soup = BeautifulSoup(full_page_source, "html.parser")

    posts = []
    for post in soup.find_all("shreddit-post"):
        post_title = post.get("post-title", "")
        author = post.get("author", "")
        comment_count = post.get("comment-count", 0)
        score = post.get("score", 0)
        posts.append(
            {
                "author": author,
                "title": post_title,
                "rating": score,
                "comments": comment_count,
            }
        )

    return posts


if __name__ == "__main__":
    chromedriver_path = get_chromedriver_path()
    service = Service(executable_path=chromedriver_path)
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(BASE_URL)
        communities_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, COMMUNITIES_BUTTON_XPATH))
        )
        communities_button.click()

        subreddit_found = find_subreddit_page(driver, SUBREDDIT_PATH)
        if not subreddit_found:
            full_url = urljoin(BASE_URL, SUBREDDIT_PATH)
            posts = scrape_subreddit_posts(driver, full_url)

            with open("posts.json", "w", encoding="utf-8") as f:
                json.dump(posts, f, indent=4, ensure_ascii=False)
            print("Данные успешно сохранены в posts.json")

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        driver.quit()
