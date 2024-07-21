import asyncio
import json
import re
from functools import lru_cache
from typing import List, Optional

from aiohttp import ClientSession, ClientTimeout
from bs4 import BeautifulSoup
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_exponential


class Settings(BaseSettings):
    mongodb_uri: str
    database_name: str
    collection_name: str
    start_url: str
    headless: bool = Field(default=True)
    user_agent: str
    max_concurrent_requests: int = Field(default=10)
    request_timeout: int = Field(default=30)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


class Book(BaseModel):
    title: str
    price: float
    stock: int
    description: Optional[str] = None
    description_url: str


@lru_cache
async def get_mongo_client():
    settings = get_settings()
    client = AsyncIOMotorClient(settings.mongodb_uri)
    try:
        yield client
    finally:
        client.close()


async def save_to_mongo(books: List[Book]):
    """Асинхронно сохраняет данные в MongoDB."""
    settings = get_settings()
    async for client in get_mongo_client():
        db = client[settings.database_name]
        collection = db[settings.collection_name]
        try:
            await collection.insert_many([book.dict() for book in books])
            logger.info(f"Записаны {len(books)} книг в MongoDB")
        except Exception as e:
            logger.error(f"Ошибка записи в MongoDB: {e}")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch(session: ClientSession, url: str) -> str:
    """Выполняет HTTP GET-запрос с повторными попытками."""
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()
    except Exception as e:
        logger.error(f"Ошибка при получении {url}: {e}")
        raise


async def scrape_category_page(
    session: ClientSession, category_url: str, semaphore: asyncio.Semaphore
) -> List[Book]:
    """Парсит одну страницу категории и возвращает список книг."""
    settings = get_settings()
    async with semaphore:
        try:
            page_content = await fetch(session, category_url)
            soup = BeautifulSoup(page_content, "html.parser")

            books = []
            for book_elem in soup.select("article.product_pod"):
                title = book_elem.select_one("h3 a")["title"]
                price = book_elem.select_one("p.price_color").text.strip()
                price = float(price.strip("£"))

                relative_url = book_elem.h3.a["href"]
                clean_url = re.sub(r"^(\.\.\/)*", "", relative_url)
                description_url = f"{settings.start_url}catalogue/{clean_url}"

                # Загружаем страницу книги для получения Availability:
                book_content = await fetch(session, description_url)
                book_soup = BeautifulSoup(book_content, "html.parser")

                availability_row = book_soup.find(
                    "th", string="Availability"
                ).find_parent("tr")
                availability_text = availability_row.find("td").text.strip()

                in_stock = "In stock" in availability_text
                stock_count = 0
                if in_stock:
                    # Извлекаем количество:
                    match = re.search(
                        r"In stock \((.*?) available\)", availability_text
                    )
                    if match:
                        stock_count = int(match.group(1))

                # парсим описание:
                description = book_soup.find("meta", {"name": "description"})[
                    "content"
                ].strip()

                books.append(
                    Book(
                        title=title,
                        price=price,
                        stock=stock_count,
                        description_url=description_url,
                        description=description,
                    )
                )
            return books
        except Exception as e:
            logger.error(f"Ошибка парсинга категории {category_url}: {e}")
            return []


async def scrape_category(
    session: ClientSession, category_url: str, semaphore: asyncio.Semaphore
):
    """Парсит все страницы категории."""
    all_books = []
    page_url = category_url
    while True:
        print(f"Парсим: {page_url}")
        books = await scrape_category_page(session, page_url, semaphore)
        all_books.extend(books)

        soup = BeautifulSoup((await fetch(session, page_url)), "html.parser")
        next_page = soup.find("li", class_="next")
        if next_page:
            next_page_url = next_page.find("a")["href"]
            page_url = f"{page_url.rsplit('/', 1)[0]}/{next_page_url}"
        else:
            break
    return all_books


async def scrape_all_books() -> List[Book]:
    """Парсит все книги, переходя по категориям."""
    settings = get_settings()
    timeout = ClientTimeout(total=settings.request_timeout)
    headers = {"User-Agent": settings.user_agent}
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async with ClientSession(timeout=timeout, headers=headers) as session:
        try:
            page_content = await fetch(session, settings.start_url)
            soup = BeautifulSoup(page_content, "html.parser")

            category_tasks = []
            categories = soup.find("ul", class_="nav-list").find("ul").find_all("a")
            # categories = soup.select('ul.nav-list li ul li a')
            for category in categories:
                category_url = f"{settings.start_url}{category['href']}"
                category_tasks.append(scrape_category(session, category_url, semaphore))

            results = await asyncio.gather(*category_tasks)
            all_books = [book for category_books in results for book in category_books]
            return all_books
        except Exception as e:
            logger.error(f"Ошибка в scrape_all_books: {e}")
            return []


async def main():
    logger.add("scraper.log", rotation="500 MB")
    logger.info("Запускаем процесс парсинга")

    books = await scrape_all_books()

    if books:
        await save_to_mongo(books)
        with open("books.json", "w", encoding="utf-8") as file:
            json.dump(
                [book.dict() for book in books], file, indent=4, ensure_ascii=False
            )

        logger.info(
            f"Парсинг завершен. Спарсено книг: {len(books)} шт. Данные записаны в MongoDB и JSON."
        )
    else:
        logger.warning("Ни одна книга не была спарсена.")


if __name__ == "__main__":
    asyncio.run(main())
