import asyncio
import argparse
from typing import List, TypedDict, Optional
from dataclasses import dataclass
import aiohttp
from aiohttp import ClientSession
from dotenv import find_dotenv, load_dotenv
import logging
from functools import lru_cache
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_fixed

# Загрузка переменных окружения из .env файла
load_dotenv(find_dotenv(".env"))


class Settings(BaseSettings):
    api_key: str = Field(..., env="api_key")
    default_location: str = "New York, NY"
    default_limit: int = 10

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


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def get_foursquare_venues(
    session: ClientSession, category: str, location: str, limit: int
) -> List[dict]:
    """Асинхронно получает список заведений от Foursquare API v3."""
    url = "https://api.foursquare.com/v3/places/search"
    headers = {"Authorization": settings.api_key, "Accept": "application/json"}
    params = {
        "query": category,
        "near": location,
        "limit": limit,
    }
    try:
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            if "results" in data:
                return data["results"]
            else:
                logger.error("Некорректный формат ответа от API")
                return []
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка при выполнении запроса к Foursquare API: {e}")
        raise


def print_venue_details(venues: List[Venue]):
    """Выводит детали заведений в консоль."""
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
    async with aiohttp.ClientSession() as session:
        venues_data = await get_foursquare_venues(session, category, location, limit)
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

    try:
        venues = get_cached_venues(category, location, args.limit)
        if venues:
            print_venue_details(venues)
        else:
            print("Заведения не найдены.")
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
