import scrapy


class ExampleSpider(scrapy.Spider):
    name = "example"
    allowed_domains = ["postnl.nl"]
    start_urls = ["https://postnl.nl"]

    def parse(self, response):
        pass
