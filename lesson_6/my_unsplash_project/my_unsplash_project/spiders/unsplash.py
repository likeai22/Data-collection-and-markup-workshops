import scrapy
from itemloaders import ItemLoader
from itemloaders.processors import MapCompose, TakeFirst

from ..items import MyUnsplashProjectItem


class UnsplashSpider(scrapy.Spider):
    name = "unsplash"
    allowed_domains = ["unsplash.com"]
    start_urls = [
        "https://unsplash.com/s/photos/construction"
    ]  # Пример начального URL, можно изменить на любую категорию

    # rules = (
    #     Rule(
    #         LinkExtractor(
    #             allow=r'/photos/\w+',
    #             restrict_xpaths='//figure[@data-test="photo-grid-masonry-figure"]//a[@itemprop="contentUrl"]'
    #         ),
    #         callback='parse_item',
    #         follow=True
    #     ),
    # ) # таким способом мы соберем ссылки на отдельные страницы картинок,
    # но парсить с отдельной страницы картинки не так удобно, как с общей

    def parse(self, response):
        images = response.xpath('//img[@data-test="photo-grid-masonry-img"]')

        for image in images:
            loader = ItemLoader(item=MyUnsplashProjectItem(), selector=image)
            loader.default_input_processor = MapCompose(str.strip)
            loader.default_output_processor = (
                TakeFirst()
            )  # Берем первое значение из списка
            loader.add_xpath(
                "image_url", "@src", MapCompose(lambda url: url.split("?")[0])
            )  # Удаляем параметры из URL
            loader.add_xpath("title", "@alt")
            loader.add_value("category", response.url.split("/")[-1])
            yield loader.load_item()
