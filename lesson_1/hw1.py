import asyncio
import argparse
from typing import List, TypedDict, Optional
from dataclasses import dataclass
from http import HTTPStatus
from aiohttp import ClientSession, ClientResponseError
from dotenv import find_dotenv, load_dotenv
import logging
from functools import lru_cache
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings
from tenacity import (
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    AsyncRetrying,
)

# Загрузка переменных окружения из .env файла
load_dotenv(find_dotenv(".env"))


class Settings(BaseSettings):
    api_key: str = Field(..., env="api_key")
    default_location: str = "New York, NY"
    default_limit: int = 10
    foursquare_api_url: str = "https://api.foursquare.com/v3/places"  # Базовый URL API

    class Config:
        env_file = ".env"


try:
    settings = Settings()
except ValidationError as e:
    print(f"Ошибка валидации переменных окружения: {e}")
    exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VenueDetails(TypedDict):
    name: str
    address: str
    rating: Optional[float]


@dataclass
class Venue:
    name: str
    address: str
    rating: Optional[float]

    @classmethod
    def from_api_response(cls, venue: dict) -> "Venue":
        location = venue.get("location", {})
        return cls(
            name=venue.get("name", "Название не указано"),
            address=location.get("formatted_address", "Адрес не указан"),
            rating=venue.get("rating"),
        )


async def _get_foursquare_venues_data(
    session: ClientSession, category: str, location: str, limit: int
) -> dict:
    """Получает данные о заведениях от Foursquare API v3."""

    url = f"{settings.foursquare_api_url}/search"
    headers = {"Authorization": settings.api_key, "Accept": "application/json"}
    params = {
        "query": category,
        "near": location,
        "limit": limit,
    }

    async with session.get(url, params=params, headers=headers) as response:
        response.raise_for_status()
        return await response.json()


async def get_foursquare_venues(
    session: ClientSession, category: str, location: str, limit: int
) -> List[dict]:
    """
    Асинхронно получает список заведений от Foursquare API v3
    с обработкой ошибок и повторными попытками.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(ClientResponseError),
        reraise=True,
    ):
        with attempt:
            data = await _get_foursquare_venues_data(session, category, location, limit)

            if "results" not in data:
                raise ValueError("Некорректный формат ответа от API")
            return data["results"]


def print_venue_details(venues: List[Venue]):
    """Выводит детали заведений в консоль."""
    if not venues:
        print("Заведения не найдены.")
        return
    for i, venue in enumerate(venues, 1):
        print(f"{i}. {venue.name}")
        print(f"   Адрес: {venue.address}")
        print(
            f"   Рейтинг: {'{:.1f}'.format(venue.rating) if venue.rating else 'Не указан'}"
        )
        print("-" * 40)


@lru_cache(maxsize=128)
def get_cached_venues(category: str, location: str, limit: int) -> List[Venue]:
    """Получает и кэширует результаты запроса."""
    return asyncio.run(async_main(category, location, limit))


async def async_main(category: str, location: str, limit: int) -> List[Venue]:
    async with ClientSession() as session:
        try:
            venues_data = await get_foursquare_venues(
                session, category, location, limit
            )
        except ClientResponseError as e:
            if e.status == HTTPStatus.UNAUTHORIZED:
                print(
                    f"Ошибка авторизации при запросе к Foursquare API. Проверьте ваш API ключ."
                )
                logger.error(f"Ошибка авторизации при запросе к Foursquare API: {e}")
            else:
                print(f"Произошла ошибка при запросе к Foursquare API: {e}")
                logger.error(f"Произошла ошибка при запросе к Foursquare API: {e}")
            return []
        except ValueError as e:
            print(f"Получен некорректный ответ от Foursquare API: {e}")
            logger.error(f"Получен некорректный ответ от Foursquare API: {e}")
            return []

        return [Venue.from_api_response(venue) for venue in venues_data]


def main():
    parser = argparse.ArgumentParser(description="Поиск заведений через Foursquare API")
    parser.add_argument("--category", type=str, help="Категория заведений")
    parser.add_argument(
        "--location",
        type=str,
        default=settings.default_location,
        help="Местоположение для поиска",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=settings.default_limit,
        help="Количество результатов",
    )
    args = parser.parse_args()

    category = args.category or input("Введите категорию заведений: ")
    location = args.location

    venues = get_cached_venues(category, location, args.limit)
    print_venue_details(venues)


if __name__ == "__main__":
    main()
