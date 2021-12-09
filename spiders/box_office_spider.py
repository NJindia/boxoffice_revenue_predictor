from os.path import exists, dirname, abspath
import re
from urllib.parse import urljoin

import numpy as np
from items import BoxOfficeSpiderItem
from scrapy import Spider
from scrapy.http import Request
from tqdm import tqdm


class BoxOfficeSpider(Spider):
    name = "box_office_spider"
    allowed_domains = ["boxofficemojo.com"]
    custom_settings = {
        'LOG_FILE': f'{dirname(dirname(abspath(__file__)))}\\logs\\box_office_spider.log',
        'ITEM_PIPELINES': {'extract_data.pipelines.BoxOfficePipeline': 300}
    }

    def parse(self, response, tconst):
        try:
            release_url = urljoin(self.base_url,
                                  response.xpath('//*[@name="releasegroup-picker-navSelector"]/option[2]').attrib[
                                      'value'])
            yield Request(url=release_url, callback=self.parse_gr, cb_kwargs={'tconst': tconst}, dont_filter=True,
                          headers=[('User-Agent', 'Mozilla/5.0')])
        except KeyError:
            item = BoxOfficeSpiderItem()
            item['tconst'] = tconst
            item['elements'] = "RELEASE_GROUP"
            yield item

    def parse_gr(self, response, tconst):
        try:
            regions = response.xpath('//*[@name="release-picker-navSelector"]/option/text()').extract()
            # Only looking for movies released domestically
            try:
                index = regions.index('Domestic') + 1
            except:
                index = -1
            if index == -1:
                item = BoxOfficeSpiderItem()
                item['tconst'] = tconst
                item['elements'] = "NO_DOMESTIC_REGION"
                yield item
            domestic_url = urljoin(self.base_url,
                                   response.xpath(f'//*[@name="release-picker-navSelector"]/option[{index}]').attrib[
                                       'value'])
            yield Request(url=domestic_url, callback=self.parse_rl, cb_kwargs={'tconst': tconst}, dont_filter=True,
                          headers=[('User-Agent', 'Mozilla/5.0')])
        except KeyError:
            item = BoxOfficeSpiderItem()
            item['tconst'] = tconst
            item['elements'] = "RELEASE_PICKER"
            yield item

    def parse_rl(self, response, tconst):
        elements = [' '.join(div.xpath('./span[1]/text()')[0].extract().split()) for div in
                    response.xpath('//*[@id="a-page"]/main/div/div[3]/div[4]/div')[0:]]
        required = ['Distributor', 'Opening', 'Budget']  # Release Date also required, see below
        item = BoxOfficeSpiderItem()
        item['tconst'] = tconst
        item["title"] = response.xpath('//*[@id="a-page"]/main/div/div[1]/div[1]/div/div/div[2]/h1/text()')[0].extract()
        if set(required).issubset(elements) and 'Release Date' in '\t'.join(elements):
            # CODE TAKEN FROM https://github.com/yjeong5126/scraping_boxofficemojo/blob/master/boxofficeinfo_spider.py
            item['domestic_revenue'] = \
            response.xpath('//*[@id="a-page"]/main/div/div[3]/div[1]/div/div[1]/span[2]/span/text()')[0].extract()

            # Distributor
            index = elements.index('Distributor') + 1
            loc_dist = f'//*[@id="a-page"]/main/div/div[3]/div[4]/div[{index}]/span[2]/text()'
            item["distributor"] = response.xpath(loc_dist)[0].extract()

            # Opening Revenue
            index = elements.index('Opening') + 1
            loc_open_rev = f'//*[@id="a-page"]/main/div/div[3]/div[4]/div[{index}]/span[2]/span/text()'
            item["opening_revenue"] = response.xpath(loc_open_rev)[0].extract()

            # Opening Theaters
            index = elements.index('Opening') + 1
            loc_open_theater = f'//*[@id="a-page"]/main/div/div[3]/div[4]/div[{index}]/span[2]/text()'
            item["opening_theaters"] = response.xpath(loc_open_theater)[0].extract().split()[0]

            # Budget
            index = elements.index('Budget') + 1
            loc_budget = f'//*[@id="a-page"]/main/div/div[3]/div[4]/div[{index}]/span[2]/span/text()'
            item["budget"] = response.xpath(loc_budget)[0].extract()

            # Release Date
            index = [idx + 1 for idx, s in enumerate(elements) if 'Release Date' in s][0]
            loc_release_date = f'//*[@id="a-page"]/main/div/div[3]/div[4]/div[{index}]/span[2]/a/text()'
            item["release_date"] = response.xpath(loc_release_date)[0].extract()

            # MPAA
            if 'MPAA' in elements:
                index = elements.index('MPAA') + 1
                loc_mpaa = f'//*[@id="a-page"]/main/div/div[3]/div[4]/div[{index}]/span[2]/text()'
                item["mpaa"] = response.xpath(loc_mpaa)[0].extract()
            else:
                item['mpaa'] = "\\N"

            # Genres
            if 'Genres' in elements:
                index = elements.index('Genres') + 1
                loc_genres = f'//*[@id="a-page"]/main/div/div[3]/div[4]/div[{index}]/span[2]/text()'
                item["genres"] = ",".join(response.xpath(loc_genres)[0].extract().split())
            else:
                item["genres"] = "\\N"
        else:
            item['elements'] = elements
        yield item

    def __get_url(self, tconst):
        url = self.base_url + "title/" + tconst + "/?ref_=bo_se_r_1"
        return url

    def start_requests(self):
        tconsts_raw = self.df['tconst'].to_numpy()
        print("Finding Valid Titles", end=':')
        tconsts = np.setdiff1d(tconsts_raw, self.errors)
        print(len(tconsts))
        tuples = [(self.__get_url(tconst), tconst) for tconst in tconsts]
        # yield Request(url=tuple[0][0], callback=self.parse, cb_kwargs={'tconst': tuple[0][1]}, dont_filter=True, headers=[('User-Agent', 'Mozilla/5.0')])
        for url, tconst in tqdm(tuples, desc="BoxOfficeMojo Scraping"):
            yield Request(url=url, callback=self.parse, cb_kwargs={'tconst': tconst}, dont_filter=True,
                          headers=[('User-Agent', 'Mozilla/5.0')])

    # TODO cmd arg to fresh scrape
    def __init__(self, df):
        self.df = df
        self.parent_path = dirname(dirname(abspath(__file__)))
        self.base_url = "https://www.boxofficemojo.com/"
        open(self.parent_path+'/logs/box_office_spider.log', 'wb').close()
        err_path = self.parent_path+"/data/errors.txt"
        if not exists(err_path): open(err_path, "w").close()
        with open(err_path, "r") as f:
            self.errors = np.array([e for e in re.findall("tt[0-9]*", f.read())])
            self.errors = np.unique(self.errors)
        print(f"Valid IMDB titles: {df.shape[0]}")
        print(f"Invalid titles: {len(self.errors)}")
