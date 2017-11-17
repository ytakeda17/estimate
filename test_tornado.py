# coding:utf-8
import io
import os
import sys
import numpy as np
import estimate
from datetime import date
from PIL import Image
import tornado.escape
import tornado.ioloop
import tornado.web

img_dir = "./tmp/img/"

class PostDebugHandler(tornado.web.RequestHandler):
    def post(self):
        for i, data_ in enumerate(self.request.files['file']):
            data = data_['body']
            fn = data_['filename']
            img = Image.open(io.BytesIO(data))
            img_path = img_dir+fn
            img.save(img_path)
        res = estimate.images2result(img_dir)

        for path in os.listdir(img_dir):
            os.remove(img_dir+path)
        self.write({"result":res})



application = tornado.web.Application([
    (r"/debug", PostDebugHandler)
])

if __name__ == "__main__":
    application.listen(5000)
    tornado.ioloop.IOLoop.instance().start()
