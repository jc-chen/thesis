import os, random
import sys
from shutil import copy2


m = sys.argv[1]
n = sys.argv[2]
dest = "/home/jean/Documents/distribution_set/" + m + "/"

for i in range(int(n)):
	src = "/home/jean/Documents/cleaned/" + random.choice(os.listdir("/home/jean/Documents/cleaned/"))
	
	copy2(src,dest)

