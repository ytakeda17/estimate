# coding:utf-8
import io
import os
import sys
import numpy as np
import estimate as es
from datetime import date
from PIL import Image
import tornado.escape
import tornado.ioloop
import tornado.web
import cv2
from itertools import chain


op_path = "../Chainer_Realtime_Multi-Person_Pose_Estimation"
sys.path.append(op_path)

import pose_detector as op
img_dir = "./tmp/img/"

def people2result(people):
    valid = 1 if len(people)>0 else 0
    scores = []
    for person in people:
        pose = np.array(person['pose_keypoints'])
        score = pose2score(pose)
        scores.append(score)
        if len(scores)>0:
            all_score = top_size_score_mean(scores)
        else:
            all_score = 0
            file_ID = re.sub( r'_keypoints.json', "", json_path.split("/")[-1])
            res[file_ID]=round(all_score*80+20*valid)
            #print([file_ID,scores])
        

pdm = op.PoseDetector('posenet', op_path+'/models/coco_posenet.npz', device=0)
class PostDebugHandler(tornado.web.RequestHandler):
    def post(self):
        res = {}
        for i, data_ in enumerate(self.request.files['file']):
            data = data_['body']
            fn = data_['filename']
            _img = Image.open(io.BytesIO(data))
            img_path = img_dir+fn
            _img.save(img_path)
            img = cv2.imread(img_path)
            people = pdm(img)
            scores = []
            valid = 1 if len(people)>0 else 0
            for person in people:
                pose = np.array(list(chain.from_iterable(person)))
                score = es.pose2score(pose)
                scores.append(score)
                if len(scores)>0:
                    all_score = es.top_size_score_mean(scores)
                else:
                    all_score = 0
            res[fn] = round(all_score*80+20*valid)
            
        for path in os.listdir(img_dir):
            os.remove(img_dir+path)
        self.write({"result":res})



application = tornado.web.Application([
    (r"/debug", PostDebugHandler)
])

if __name__ == "__main__":
    application.listen(6000)
    tornado.ioloop.IOLoop.instance().start()
