# Tag-Recognizer
This repo contains code for recognizing Hospital Tags in Canadian Hospitals. 

## Objective 

The objective of this project is to build an automatic recognizer of hospital tags using Optical Character Recognition (OCR) and Support Vector Machines (SVM).

## Description

Images of hospital tags are pre-processed. The tag’s entities are localized. The localized entities are extracted and fed into OCR. The outputs of all the entities are bundled into a JSON object as output. Python API is created for the developers to consume this tag recognizer service. 

## Input & Output 

Input:  Images of Hospital Tags from North American Hospitals

Output: JSON object of recognized characters

## Tools Used

Python, Scikit-learn, Tesseract, Cherrypy


## Technical Report

Find the report [here](https://github.com/subashgandyer/Tag-Recognizer/blob/master/TagMe-Technical%20Report.pdf)
