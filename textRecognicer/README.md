
# this program recognises hand writting. 
It is an attempt to recognice the text on the transponders
It has trouble with noise in the images, since the letters are not on a white background.

This part of the program is under development

#To train the classifier:
python train.py --dataset data/digits.csv --model models/svm


# to recognice numbers in a picture
python classify.py --model models/svm.cpickle --image images/B21_A170.png


# dependencies
sklearn
mahotas
opencv 2.4.9
imutils
python 2.7