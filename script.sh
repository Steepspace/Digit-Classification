#!/usr/bin/env bash
# naive bayes digits
python dataClassifier.py -c naiveBayes -t 1000 -s 1000 -k 0.05 > out.txt
mv results.txt results/nb_digits.csv

# naive bayes faces
python dataClassifier.py -c naiveBayes -k 0.001 -d faces > out.txt
mv results.txt results/nb_faces.csv

# perceptron digits
python dataClassifier.py -c perceptron -i 5 -t 1000 -s 1000 > out.txt
mv results.txt results/percp_digits.csv

# perceptron faces
python dataClassifier.py -c perceptron -i 5 -d faces > out.txt
mv results.txt results/percp_faces.csv

# mira digits
python dataClassifier.py -c mira -i 5 -t 1000 -s 1000 > out.txt
mv results.txt results/mira_digits.csv

# mira faces
python dataClassifier.py -c mira -i 5 -d faces > out.txt
mv results.txt results/mira_faces.csv
