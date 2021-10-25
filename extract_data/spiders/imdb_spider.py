import pickle
from os.path import dirname, abspath
from pydispatch import dispatcher
from scrapy import Spider
from scrapy import signals
from scrapy.http import Request
from tqdm import tqdm


class IMDBSpider(Spider):
    name = "imdb_spider"
    allowed_domains = ["imdb.com"]
    custom_settings = {
        'LOG_FILE': f'{dirname(dirname(abspath(__file__)))}\\logs\\imdb_spider.log',
    }

    def parse(self, response, tconst):
        try:
            rating = \
                response.xpath(f'//*[@href="/title/{tconst}/parentalguide/certificates?ref_=tt_ov_pg"]/text()').extract()[0].upper()
        except:
            rating = 'UNRATED'
        ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17']
        if rating not in ratings:
            # X is equivalent to NC-17, which is currently used
            if rating == 'X':
                rating = 'NC-17'
            else:
                rating = 'UNRATED'
        self.df.loc[self.df['tconst'] == tconst, 'mpaa'] = rating

    @staticmethod
    def __get_url(tconst):
        return "https://www.imdb.com/title/" + tconst + "/"

    def start_requests(self):
        tuples = [(self.__get_url(tconst), tconst) for tconst in self.df['tconst']]
        for url, tconst in tqdm(tuples, desc="IMDB Scraping"):
            yield Request(url=url, callback=self.parse, cb_kwargs={'tconst': tconst}, dont_filter=True,
                          headers=[('User-Agent', 'Mozilla/5.0')])

    def spider_closed(self, spider):
        with open(self.parent_path+'/pickled_data/mpaa_data.pickle', 'wb') as f:
            pickle.dump(self.df, f)

    def __init__(self, df, **kwargs):
        super().__init__(**kwargs)
        self.df = df.loc[df['mpaa'] == '\\N', ['tconst', 'mpaa']]
        self.parent_path = dirname(dirname(abspath(__file__)))
        open(self.parent_path+'/logs/imdb_spider.log', 'wb').close()
        dispatcher.connect(self.spider_closed, signal=signals.spider_closed)
