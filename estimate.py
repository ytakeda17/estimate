import json
import os,sys,re
import numpy as np
import glob
from scipy.stats import gmean
import subprocess

def get_jsons(pics_dir, jsons_dir):
   subprocess.call(["./openpose.bin", "--image_dir", pics_dir, "--write_keypoint_json", jsons_dir, "--no_display", "--num_gpu",  "1", "--keypoint_scale", "3", "--model_folder", "./models/"])

def cosine(u,v):
   return (abs(np.dot(u,v))/(np.linalg.norm(u)*np.linalg.norm(v)))

def top_width_score_mean(ary):
   sorted_ary = sorted(ary,reverse=True,key=lambda x:x[1])
   selected = []
   top_w = sorted_ary[0][1]
   threshold = top_w/3
   
   for v,w,h in sorted_ary:
      if (top_w - w) < threshold:
         selected.append(v*min([max([h*1.25,w*2]), 1]))
         top_w = w      
      else:
         break
   return np.mean(selected)


def estimate(images_path):
   #argv = sys.argv
   #pics_path = argv[1]
   #jsons_path = argv[2].rstrip("/")
   jsons_path = "tmp/out"
   get_jsons(images_path, jsons_path)
   jsons_path_format = jsons_path+"/*.json"

   res = {}
   valid = 0
   for json_path in glob.glob(jsons_path_format):
      all_score = 0
      with open(json_path,"r") as f:
         data = json.load(f)
         people = data["people"]
         valid = 1 if len(people)>0 else 0
         scores = []
         for person in people:
            score = 0
            pose = np.array(person['pose_keypoints'])
            ys = np.array([x for x in pose[list(range(1,54,3))] if x >0.001])
            xs = np.array([x for x in pose[list(range(0,54,3))] if x >0.001])
            height = max(ys) - min(ys)
            width = max(xs) -min(xs)
            nos = pose[0*3:1*3]
            rye = pose[14*3:15*3]
            lye = pose[15*3:16*3]
            rar = pose[16*3:17*3]
            lar = pose[17*3:18*3]

            lye_line = nos[:2]-lye[:2]
            rye_line = nos[:2]-rye[:2]
            lar_line = lye[:2]-lar[:2]
            rar_line = rye[:2]-rar[:2]
            
            possibilities = [x[2] for x in  [rye,lye,rar,lar]]

            if min(possibilities) > 0.01:
               rdis = ((rye[0]-rar[0])**2 + (rye[1]-rar[1])**2)**0.5
               ldis = ((lye[0]-lar[0])**2 + (lye[1]-lar[1])**2)**0.5
               score = np.mean([cosine(lye_line,lar_line),cosine(rye_line,rar_line)])
               score *= (1-np.min([rdis/ldis,ldis/rdis]))
            else:
               score = 1
            scores.append((score,width,height))
         if len(scores)>0:
            all_score = top_width_score_mean(scores)
         else:
            all_score = 0
      file_ID = re.sub( r'_keypoints.json', "", json_path.split("/")[-1])
      res[file_ID]=round(all_score*80+20*valid)
      #print(scores)
      #print([json_path.split("/")[-1],round(all_score*80+20*valid)])
   for path in os.listdir(jsons_path):
      os.remove(jsons_path+"/"+path)
   print(res)
   return res
   
