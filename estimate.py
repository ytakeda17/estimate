import json
import os,sys
import numpy as np
import glob

def get_jsons(pics_dir, jsons_dir):
   pass



if __name__=="__main__":
   argv = sys.argv
   pics_path = argv[1]
   jsons_path = argv[2].rstrip("/")
   
   get_jsons(pics_path, jsons_path)
   jsons_path_format = jsons_path+"/*.json"
   for json_path in glob.glob(jsons_path_format):
      score = 0
      with open(json_path,"r") as f:
         data = json.load(f)
         people = data["people"]
         rates = []
         for person in people:
            pose = person['pose_keypoints']
            rye = pose[14*3:15*3]
            lye = pose[15*3:16*3]
            rar = pose[16*3:17*3]
            lar = pose[17*3:18*3]
            possibilities = [x[2] for x in  [rye,lye,rar,lar]]
            if np.max(possibilities)>0:
               if len([x for x in possibilities if x > 0.1])== 4:
                  rdis = ((rye[0]-rar[0])**2 + (rye[1]-rar[1])**2)**0.5
                  ldis = ((lye[0]-lar[0])**2 + (lye[1]-lar[1])**2)**0.5
                  print([rdis,ldis])
                  rates.append(rdis/ldis)
               else:
                  rates.append(1)
         ave_rate = np.mean(rates)
      score += 1-ave_rate
      print([score, json_path])
