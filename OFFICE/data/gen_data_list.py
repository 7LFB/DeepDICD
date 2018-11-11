# genenrate data list
import os
import numpy as np 
import glob
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=int, default=0)

args=parser.parse_args()

DIR='/home/comp/chongyin/DataSets/DA/Office-31/Original_images'
FOLDER=['amazon','dslr','webcam']
d=args.domain

f=open('./list/'+FOLDER[d]+'.txt','w')
for dirs,flds,_ in os.walk(os.path.join(DIR,FOLDER[d],'images')):
	for index,folder in enumerate(flds):
		print(index,folder)
		for _,_,files in os.walk(os.path.join(dirs,folder)):
			for file in files:
				name=FOLDER[d]+'/images/'+folder+'/'+file+' '+str(index)+'\n'
				f.write(name)

f.close()

