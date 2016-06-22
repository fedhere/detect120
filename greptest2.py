from __future__ import print_function

__author__ = 'Mohit Sharma'

import requests
from bs4 import BeautifulSoup as bs
import urllib2
import os

_URL = 'http://serv.cusp.nyu.edu/files/high_speed/'

r = requests.get(_URL)
soup = bs(r.text)
urls = []
names = []

print('Populating the list of files.. This can take sometime')
for i, link in enumerate(soup.findAll('a')):
    _fullURL = _URL + link.get('href')
    if _fullURL.endswith('.raw'):
        urls.append(_fullURL)
        names.append(soup.select('a')[i].attrs['href'])
        print(names)

names_urls = zip(names, urls)

print('Grabbing all the files..')
# TBD: Parallelize it..!                                                                                                                                                                                                                      
for name, url in names_urls:
    print(url)
    req = urllib2.Request(url)
    res = urllib2.urlopen(req)
    raw = open(os.getcwd()+'/'+name,'wb')
    raw.write(res.read())
    raw.close()
