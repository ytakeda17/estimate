import json
import os,sys
import numpy as np
import glob
from scipy.stats import gmean

def get_jsons(pics_dir, jsons_dir):
   pass

def cosine(u,v):
   #return min((1-abs(np.dot(u,v))/(np.linalg.norm(u)*np.linalg.norm(v)))*(2**(0.5)), 1)
   return (1-abs(np.dot(u,v))/(np.linalg.norm(u)*np.linalg.norm(v)))*(2**(0.5))

def top_mean(ary,threshold=0.2):
   sorted_ary = sorted(ary,reverse=True)
   selected = []
   top = sorted_ary[0]
   for element in sorted_ary:
      if top-element < threshold:
         selected.append(element)
         top = element
      else:
         break
   return np.mean(selected)

if __name__=="__main__":
   argv = sys.argv
   pics_path = argv[1]
   jsons_path = argv[2].rstrip("/")
   get_jsons(pics_path, jsons_path)
   jsons_path_format = jsons_path+"/*.json"
   #print(os.getcwd())
   res = {}
   for json_path in glob.glob(jsons_path_format):
      all_score = 0
      with open(json_path,"r") as f:
         data = json.load(f)
         people = data["people"]
         scores = []
         heights = []
         for person in people:
            score = 0
            pose = np.array(person['pose_keypoints'])
            ys = np.array([x for x in pose[list(range(1,54,3))] if x >0.01])
            height = max(ys) - min(ys)
            heights.append(height)
            rsh = pose[2*3:3*3]
            lsh = pose[5*3:6*3]
            rye = pose[14*3:15*3]
            lye = pose[15*3:16*3]
            rar = pose[16*3:17*3]
            lar = pose[17*3:18*3]

            shoulder_line = rsh[:2]-lsh[:2]
            l_line = lye[:2]-lar[:2]
            r_line = rye[:2]-rar[:2]
            
            possibilities = [x[2] for x in  [rye,lye,rar,lar]]
            if max(possibilities)>0.5:
               if min(possibilities) > 0.1:
                  rdis = ((rye[0]-rar[0])**2 + (rye[1]-rar[1])**2)**0.5
                  ldis = ((lye[0]-lar[0])**2 + (lye[1]-lar[1])**2)**0.5
                  score = np.mean([cosine(l_line,shoulder_line),cosine(r_line,shoulder_line)])
                  #rates.append(np.min([rdis/ldis,ldis/rdis]))
                  score = np.min([rdis/ldis,ldis/rdis])
               else:
                  #rates.append(0)
                  score = 1
            else:
               score = 0
            scores.append(score*np.min([height/0.8,1]))
         if len(scores)>0:
            #ave_rate = gmean(rates)
            #all_score = gmean(scores)#*max(heights)
            #all_score = np.mean(scores)#*max(heights)
            all_score = top_mean(scores)#*max(heights)
         else:
            all_score = 0
      #print(scores)
      #print(heights)
      res[json_path.split("/")[-1]]=all_score
      print([json_path.split("/")[-1],round(all_score*100)])
   #print(sorted(res))
