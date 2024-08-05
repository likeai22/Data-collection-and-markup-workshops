import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from urllib.parse import urljoin
from ..items import MyUnsplashProjectItem


# class UnsplashSpider(CrawlSpider):
#     name = "unsplash"
#     allowed_domains = ["www.unsplash.com"]
#     start_urls = ["https://www.unsplash.com/s/photos/galaxy"]
#     rules = (
#         Rule(
#             LinkExtractor(restrict_xpaths="img[starts-with(@class,'I7OuT')]"),
#             callback="parse_item",
#             follow=True
#         ),
#     )
#
#     def parse_item(self, response):
#
#         item = MyUnsplashProjectItem()
#         image_url = response.xpath('//img[@itemprop="contentUrl"]/@src').get()
#         item["image_urls"] = [urljoin(response.url, image_url)]
#         item["title"] = response.xpath('//h1/text()"]').get()
#         item["category"] = response.url.split("/photos/")[-1].split("/")[0]
#         print("item", item)
#         yield item


class UnsplashSpider(scrapy.Spider):
    name = "unsplash"
    allowed_domains = ["unsplash.com"]
    start_urls = ["https://unsplash.com/s/photos/galaxy"]

    def parse(self, response):
        for img in response.xpath('//figure//img[starts-with(@class, "I7OuT")]'):
            item = MyUnsplashProjectItem()
            image_url = img.xpath("./@src").get()
            print("image_url", image_url)
            # item["image_urls"] = [urljoin(response.url, image_url)]
            # item["title"] = img.xpath('./@alt').get()
            # item["category"] = "galaxy"
            # print("item", item)
            yield item