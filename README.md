# POS_Tagging_and_Named_Entity_Recognition_on_the_American_State_of_the_Union_Addresses

General Information:
---------
* State of the Union (SOTU) is a major speech traditionally given by the President of the U.S. at the beginning of each calendar year.
* The president is addressing both houses (Congress and Senate) - reflecting on the achievements (and failures) of the passing year and outlining policy priorities for the coming year. 
* Content and style of the SOTU addresses vary along the years and depend on the shifting culture, the personality of the president and the challenges of the time. 

This Project:
---------
* This project focuses on comparing the distribution of POS and NE tags over time and presidents, in the U.S State of the Union Addresses.
The Python NLTK package is used for executing tagging modules (POS, NER) on the attached corpus.

Instructions:
---------
NLTK provides NER and POS tagging interfaces. In order to use the NLTK taggers the NLTK package has to be installed:
```python
pip install nltk 
```
and then download the nessecesary corpora:
```python
import nltk
nltk.download()
```
