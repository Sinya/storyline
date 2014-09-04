#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import re
from goose import Goose
from goose.text import StopWordsChinese


resquest = []
page_start = 0
count = 0 # test

for i in range(10):

	# ===========================================
	# url on search result

	resquest.append(requests.get("https://www.google.com.tw/search?q=%E9%BB%83%E8%89%B2%E5%B0%8F%E9%B4%A8+site:http://news.pts.org.tw&lr=&hl=zh-TW&as_qdr=all&ei=blL7U-L2EIvd8AWzvoKAAg&start=" + str(page_start) +"&sa=N&biw=1440&bih=779").text)
	page_start += 0

	res = str(resquest)
	pos = [m.start() for m in re.finditer('<cite>',res)]


	if pos > 0:
		for s in range(0, len(pos)-1):
			urlPos = res[pos[s]:pos[s+1]].find('</cite>')
			url = res[pos[s]+6:pos[s]+urlPos].encode('utf-8')

			if url[0:15] == 'news.pts.org.tw':
				print url



		# ===========================================
		# news text in the url

				inner_res = []
				# inner_res.append(requests.get('http://'+url).text)
				content = ""


				try:

					g = Goose({'stopwords_class': StopWordsChinese})
					article = g.extract(url='http://'+url)


					# print article.title.encode('utf-8')
					# print article.meta_description.encode('utf-8')
					# print article.cleaned_text[:].encode('utf-8')
					# if article.top_image.src > 0:
					# 	print article.top_image.src
					# content += newsUrl + "\n"

					if article.title is None:
						pass
					else:
						content += article.title.encode('utf-8') + "\n" 

					if article.meta_description is None:
						pass
					else:
						content += article.meta_description.encode('utf-8') + "\n"

					if article.cleaned_text[:] is None:
						pass
					else:
						content += article.cleaned_text[:].encode('utf-8') + "\n"

					if article.top_image is None:
						pass
					else:
						content += article.top_image.src + "\n"

					if article.movies is None:
						pass
					else:
						print "Hi :D movie"
						for movie in article.movies:
							print article.movies[0].src[2:]
							print article.movies[0].embed_code
							print article.movies[0].embed_type
							print article.movies[0].width
							print article.movies[0].height


					# content += "\n"
					# print inner_res
					content += str('http://'+url)
					print content

					# with open( date[0:6] + "appleURL.txt", "a") as myfile:
					# 	myfile.write(content)
					# 	myfile.close()

					count += 1
					print count

				except:
				    print("except!!!")
				 
				print("after exception....")


		# ===========================================
		# create table


		# import pandas as pd
		# news_table = pd.DataFrame(p)
		# news_table.columns = ['title','date','kind','url']

		# print news_table
