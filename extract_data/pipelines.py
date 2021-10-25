# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from pydispatch import dispatcher
import pandas as pd
from scrapy import signals
import csv
import pickle


class BoxOfficePipeline:
    def __init__(self):
        columns = ['tconst', 'title', 'distributor', 'domestic_revenue', 'opening_revenue', 'opening_theaters',
                   'budget', 'mpaa', 'genres', 'release_date']
        self.df = pd.DataFrame(columns=columns)
        self.errors = []
        dispatcher.connect(self.spider_closed, signal=signals.spider_closed)

    def spider_closed(self, spider):
        with open("data/errors.txt", "a") as f:
            f.writelines(self.errors)
        with open('pickled_data/box_office_data.pickle', 'wb') as f:
            pickle.dump(self.df, f)

    def process_item(self, item, spider):
        # If elements has been scraped, not enough elements exist
        try:
            if item['elements'] == "RELEASE_GROUP":
                self.errors.append(f"{item['tconst']}: RELEASE GROUP\n")
            elif item['elements'] == "RELEASE_PICKER":
                self.errors.append(f"{item['tconst']}: RELEASE PICKER\n")
            else:
                self.errors.append(f"{item['tconst']}: {item['elements']}\n")
            return
        except KeyError:
            pass
        d = {
            'tconst': [item['tconst']],
            'title': [item['title']],
            'distributor': [item['distributor']],
            'domestic_revenue': [item['domestic_revenue']],
            'opening_revenue': [item['opening_revenue']],
            'opening_theaters': [item['opening_theaters']],
            'budget': [item['budget']],
            'mpaa': [item['mpaa']],
            'genres': [item['genres']],
            'release_date': [item['release_date']]
        }
        df = pd.DataFrame.from_dict(d)
        self.df = self.df.append(df)
        return item
