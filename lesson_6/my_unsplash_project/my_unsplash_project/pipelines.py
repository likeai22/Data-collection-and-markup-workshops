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
    def __init__(self, filename="unsplash_images.csv", hash_method=hashlib.sha1):
        self.filename = filename
        self.hash_method = hash_method
        self.processed_items = set()

    def open_spider(self, spider):
        try:
            self.file = open(self.filename, "wb")
            self.exporter = CsvItemExporter(self.file)
            self.exporter.start_exporting()
        except IOError as e:
            logging.error(f"Ошибка открытия файла: {e}")
            raise

    def close_spider(self, spider):
        try:
            self.exporter.finish_exporting()
            self.file.close()
        except Exception as e:
            logging.error(f"Ошибка закрытия файла: {e}")

    def process_item(self, item, spider):
        item_id = self.hash_method(str(item).encode()).hexdigest()
        if item_id not in self.processed_items:
            self.processed_items.add(item_id)
            logging.info(f"Processing item: {item}")
            self.exporter.export_item(item)
        else:
            logging.info(f"Пропуск повторяющегося элемента: {item}")
        return item
