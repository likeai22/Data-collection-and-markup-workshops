# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import csv

# useful for handling different item types with a single interface
import scrapy
from scrapy.exporters import CsvItemExporter
from scrapy.pipelines.images import ImagesPipeline


class MyUnsplashProjectPipeline:
    def get_media_requests(self, item, info):
        return [
            scrapy.Request(x, meta={"item": item}) for x in item.get("image_urls", [])
        ]

    def item_completed(self, results, item, info):
        if "images" in item:
            for ok, x in results:
                if ok:
                    item["images"] = x["path"]
        return item


class CSVPipeline(object):
    def open_spider(self, spider):
        self.file = open("unsplash_images.csv", "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["image_url", "local_path", "title", "category"])

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        self.writer.writerow(
            [
                item["image_urls"][0],
                item["images"],
                item["title"],
                item["category"],
            ]
        )
        return item
