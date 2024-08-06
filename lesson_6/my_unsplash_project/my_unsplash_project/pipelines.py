# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import hashlib
import scrapy
from scrapy.exporters import CsvItemExporter
from scrapy.pipelines.images import ImagesPipeline

import logging


class MyUnsplashProjectPipeline:
    def process_item(self, item, spider):
        return item


class CustomImagesPipeline(ImagesPipeline):
    def file_path(self, request, response=None, info=None, *, item=None):
        image_guid = hashlib.sha1(request.url.encode()).hexdigest()
        return f"{item['title']} - {image_guid}.jpg"

    def get_media_requests(self, item, info):
        yield scrapy.Request(item["image_url"])

    def item_completed(self, results, item, info):
        if results:
            item["images"] = [x["path"] for ok, x in results if ok]
        return item


class CSVPipeline(object):
    def open_spider(self, spider):
        self.file = open("unsplash_images.csv", "wb")
        self.exporter = CsvItemExporter(self.file)
        self.exporter.start_exporting()

    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.file.close()

    def process_item(self, item, spider):
        logging.info(f"Processing item: {item}")
        self.exporter.export_item(item)
        return item
