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

def sine(u,v):
   return (1-cosine(u,v)**2)**(0.5)

def dist(u,v):
   return ((u[0]-v[0])**2 + (u[1]-v[1])**2)**0.5
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

def top_size_score_mean(ary):
   sorted_ary = sorted(ary,reverse=True,key=lambda x:x[1]*x[2])
   selected = []
   top_size = sorted_ary[0][1]*sorted_ary[0][2]
   threshold = top_size*(5/9)
   
   for v,w,h in sorted_ary:
      if  w * h > threshold:
         selected.append(v*min([3*w*h, 1]))
      else:
         break
   return np.mean(selected)

## 体積でスコアとるやつを選別

## max(possibility)<0.5は0
## 両目両耳→長さではなくmi(cos(目と耳, 目と鼻),cos(目と耳, 目と鼻))
## 両目片耳→if max(目と鼻と鼻と首の線の角度)＞閾値: 数値; else: 1;
## 片目　→ 1

def images2result(images_path):
   jsons_path = "tmp/out"
   get_jsons(images_path, jsons_path)
   jsons_path_format = jsons_path+"/*.json"
   json_paths = glob.glob(jsons_path_format)
   res = jsons2result(json_paths)
   for path in os.listdir(jsons_path):
      os.remove(jsons_path+"/"+path)
   return res

def jsons2result(json_paths):
   res = {}
   valid = 0
   for json_path in json_paths:
      all_score = 0
      with open(json_path,"r") as f:
         data = json.load(f)
         people = data["people"]
         valid = 1 if len(people)>0 else 0
         scores = []
         for person in people:
            pose = np.array(person['pose_keypoints'])
            score = pose2score(pose)
            scores.append(score)
         if len(scores)>0:
            all_score = top_width_score_mean(scores)
         else:
            all_score = 0
      file_ID = re.sub( r'_keypoints.json', "", json_path.split("/")[-1])
      res[file_ID]=round(all_score*80+20*valid)
      print([file_ID,scores])
   return res
   

   
def pose2score(pose):
   ys = np.array([x for x in pose[list(range(1,54,3))] if x >0.001])
   xs = np.array([x for x in pose[list(range(0,54,3))] if x >0.001])
   height = max(ys) - min(ys)
   width = max(xs) -min(xs)
   nos = pose[0*3:1*3]
   nec = pose[1*3:2*3]
   rye = pose[14*3:15*3]
   lye = pose[15*3:16*3]
   rar = pose[16*3:17*3]
   lar = pose[17*3:18*3]

   lye_line = lye[:2]-nos[:2]
   rye_line = rye[:2]-nos[:2]
   lar_line = lar[:2]-lye[:2]
   rar_line = rar[:2]-rye[:2]
   nos_line = nos[:2]-nec[:2]
   
            
            
   probs = [x[2] for x in  [rye,lye,rar,lar]]
   eye_probs = [x[2] for x in  [rye,lye]]
   ear_probs = [x[2] for x in  [rar,lar]]
   score = 0
   if max(probs)>0.5:
      if min(eye_probs) > 0.01:
         if min(ear_probs) > 0.01:
            eye_nos_dis = dist((rye+lye)/2,nos)
            ear_nos_dis = dist((rar+lar)/2,nos)
            naturality = ear_nos_dis / eye_nos_dis
            score = 1 if naturality > 1 else 0.2
         else:
            if ear_probs[0]>ear_probs[1]:
               naturality = sine(lye_line,nos_line)
            else:
               naturality = sine(rye_line,nos_line)
            score = naturality if naturality < 0.3 else 1 
      else:
         score = 1
   else:
      score = 0
   return (score,width,height)





#########################################################################3
def pics_to_res_old(images_path):
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
            nec = pose[1*3:2*3]
            rye = pose[14*3:15*3]
            lye = pose[15*3:16*3]
            rar = pose[16*3:17*3]
            lar = pose[17*3:18*3]

            lye_line = lye[:2]-nos[:2]
            rye_line = rye[:2]-nos[:2]
            lar_line = lar[:2]-lye[:2]
            rar_line = rar[:2]-rye[:2]
            nos_line = nos[:2]-nec[:2]

            
            
            possibilities = [x[2] for x in  [rye,lye,rar,lar]]
            eye_pos = [x[2] for x in  [rye,lye]]
            ear_pos = [x[2] for x in  [rar,lar]]
            #print(possibilities)
            if min(eye_pos) > 0.01:
               #rdis = ((rye[0]-rar[0])**2 + (rye[1]-rar[1])**2)**0.5
               #ldis = ((lye[0]-lar[0])**2 + (lye[1]-lar[1])**2)**0.5
               
               rdis = ((rye[0]-nos[0])**2 + (rye[1]-nos[1])**2)**0.5
               ldis = ((lye[0]-nos[0])**2 + (lye[1]-nos[1])**2)**0.5
               
               #score = np.mean([cosine(lye_line,lar_line),cosine(rye_line,rar_line)])

               #score = min([sine(lye_line,nos_line),sine(rye_line,nos_line)])
               #print([rdis,ldis])
               score = (1-np.min([rdis/ldis,ldis/rdis]))
               if min(ear_pos) > 0.01:
                  score *= np.mean([cosine(lye_line,lar_line),cosine(rye_line,rar_line)])
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
   #print(res)
   return res

