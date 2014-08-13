#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import chardet

path ="fat_news/"

for root, dirs, files in os.walk(path):
    # print root
    for f in files:
        doc = ""
        date = 0
        url = ""
        title = ""

        file_path = os.path.join(root, f)
        with open( file_path ) as op:
            for i,line in enumerate(op):
                if i is 0:
                    url = line
                elif i is 2:
                    date = line
                elif i is 4:
                    title = line
                else:
                    doc += line
        fileread = lambda filename: open(filename, "rb").read()
        coding = chardet.detect(fileread(file_path))['encoding']
		# {'confidence': 0.99, 'encoding': 'utf-8'}
        doc = doc.decode( coding )
        print doc.encode('utf-8')
        print "=====url======"
        print url
        print "=====date======"
        print date
        print "======title====="
        print title


        