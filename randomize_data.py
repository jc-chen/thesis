import os, random
import sys
from shutil import copy2

'''
Randomly selects n molecule files from source directory and copies to destination directory
Usage: python randomize_data.py <source dir> <destination dir> <n>
'''

src = sys.argv[1]
dst = sys.argv[2]
n = int(sys.argv[3])

while n>0:
	file = random.choice(os.listdir(src))
	if not os.path.exists(dst+file):
		n-=1 
		copy2(src+file,dst)

