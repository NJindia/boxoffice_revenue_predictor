# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class BoxOfficeSpiderItem(scrapy.Item):
    title = scrapy.Field()
    tconst = scrapy.Field()
    domestic_revenue = scrapy.Field()
    distributor = scrapy.Field()
    opening_revenue = scrapy.Field()
    opening_theaters = scrapy.Field()
    release_date = scrapy.Field()
    budget = scrapy.Field()
    mpaa = scrapy.Field()
    genres = scrapy.Field()
    elements = scrapy.Field()
    pass
