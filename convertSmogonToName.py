import os
import sys

pFile = open(sys.argv[1], 'r')
targetFile = open(sys.argv[2], 'w')

for line in pFile.readlines():
    targetFile.write(line.split('|')[2].strip() + "\n")

targetFile.close()
