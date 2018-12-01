import os
import sys

'''
Changes molecular data in text file to scientific notation to avoid parsing issues
Usage: python process_data.py <source directory> <output directory>
'''

input_url = sys.argv[1]
output_url = sys.argv[2]

for file in os.listdir(input_url):
	with open(input_url+file,'r') as fin:
		with open(output_url+file,'w') as fout:
			for line in fin:
				fout.write(line.replace("*10^","0e"))
