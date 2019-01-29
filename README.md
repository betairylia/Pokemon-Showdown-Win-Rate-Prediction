# Pokemon Showdown Win Rate Prediction

This is an simple model predicts win-rate for pokemon (showdown) matches.  
**WORK IN PROGRESS** (including this readme)

## Requirements ( latest should work )
* python 3.6  
* tensorflow  
* tensorlayer  
* termcolor (used to print out pretty logs)

## Usage (v2)
### Get match data

`python spider.py [tierName]`  

e.g.

`python spider.py gen7vgc2019sunseries`

### Convert raw data to embedded data

`python convertRaw.py [filename] (-max maxSpecies)`

e.g.

`python convertRaw.py gen7ou.txt`

### Train the model

`python trainv2.py [formatPrefix (data file name without .txt)]`

Please use -h option to print the help message for more options.  
Note: -fmt & -load is not working now.

### Notes

Currently accuracy is about 61.9% (best performance).  
maxpool is better than summation (might for classification problems).
