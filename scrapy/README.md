https://scrapeops.io/python-scrapy-playbook/scrapy-beginners-guide/

scrapy startproject postnl
scrapy genspider example postnl.nl

## scrapy.cfg
[settings]
default = chocolatescraper.settings
shell = ipython

scrapy shell postnl.nl

settings.py is where all your project settings are contained, like activating pipelines, middlewares etc. Here you can change the delays, concurrency, and lots more things.
items.py is a model for the extracted data. You can define a custom model (like a ProductItem) that will inherit the Scrapy Item class and contain your scraped data.
pipelines.py is where the item yielded by the spider gets passed, it’s mostly used to clean the text and connect to file outputs or databases (CSV, JSON SQL, etc).
middlewares.py is useful when you want to modify how the request is made and scrapy handles the response.
scrapy.cfg is a configuration file to change some deployment settings, etc.
