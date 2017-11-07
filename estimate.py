import json
import os,sys
import numpy as np
import glob
from scipy.stats import gmean

def get_jsons(pics_dir, jsons_dir):
   pass

def cosine(u,v):
   return (abs(np.dot(u,v))/(np.linalg.norm(u)*np.linalg.norm(v)))

def top_height_score_mean(ary,threshold=0.2):
   sorted_ary = sorted(ary,reverse=True,key=lambda x:x[1])
   selected = []
   top = sorted_ary[0][1]
   for v,h in sorted_ary:
      if (top - h) < threshold:
         selected.append(v*min([h*1.25, 1]))
         top = h
      else:
         break
   return np.mean(selected)


if __name__=="__main__":
   argv = sys.argv
   pics_path = argv[1]
   jsons_path = argv[2].rstrip("/")
   get_jsons(pics_path, jsons_path)
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
            height = max(ys) - min(ys)
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
            scores.append((score,height))
         if len(scores)>0:
            all_score = top_height_score_mean(scores)
         else:
            all_score = 0
            
      res[json_path.split("/")[-1]]=all_score
      print([json_path.split("/")[-1],round(all_score*80+20*valid)])

