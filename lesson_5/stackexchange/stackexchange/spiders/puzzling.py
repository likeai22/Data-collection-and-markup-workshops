import scrapy
from scrapy.utils.response import get_base_url
from urllib.parse import urljoin

from ..items import StackexchangeItem


class PuzzlingSpider(scrapy.Spider):
    name = "puzzling"
    allowed_domains = ["puzzling.stackexchange.com"]
    start_url_base = "https://puzzling.stackexchange.com/questions/tagged/mathematics"
    start_urls = [
        f"{start_url_base}?tab=newest&page=1&pagesize=15"
    ]
    max_pages = 1
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 2,
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/58.0.3029.110 Safari/537.3",
    }

    def parse(self, response):
        base_url = get_base_url(response)
        self.logger.info(f"Parsing page: {response.url}")

        for question in response.xpath('//div[contains(@id, "question-summary")]'):
            try:
                item = StackexchangeItem(
                    title=question.xpath(
                        './/h3[@class="s-post-summary--content-title"]/a/text()'
                    ).get(),
                    link=urljoin(
                        base_url,
                        question.xpath(
                            './/h3[@class="s-post-summary--content-title"]/a/@href'
                        ).get(),
                    ),
                    votes=question.xpath(
                          './/div[contains(@class, "s-post-summary--stats-item")][1]//span[@class="s-post-summary--stats-item-number"]/text()'
                    ).get(default="0").strip(),
                    answers=question.xpath(
                        './/div[contains(@class, "s-post-summary--stats-item")][2]//span[@class="s-post-summary--stats-item-number"]/text()'
                    ).get(default="0").strip(),
                    views=question.xpath(
                        './/div[contains(@class, "s-post-summary--stats-item")][3]//span[@class="s-post-summary--stats-item-number"]/text()'
                    ).get(default="0").strip(),
                    tags=question.xpath(
                        './/a[contains(@class, "s-tag post-tag flex--item mt0")]/text()'
                    ).getall(),
                )
                yield item
            except Exception as e:
                self.logger.error(f"Error parsing question: {e}")

        current_page = int(response.url.split("page=")[1].split("&")[0])
        if current_page < self.max_pages:
            next_page = response.url.replace(
                f"page={current_page}", f"page={current_page + 1}"
            )
            yield scrapy.Request(next_page, callback=self.parse)
