import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def review_to_words(review_text):
    # remove html tags
    example1 = BeautifulSoup(review_text)
    # just letters 
    letters_only = re.sub('[^a-zA-Z]', " ", example1.get_text())
    # to words
    words = letters_only.lower().split()
    stops = set(stopwords.words('english'))
    # remove stop words
    meaningful_words = [w for w in words if not w in stops]
    # join words altoghter again
    return " ".join(meaningful_words)

if __name__ == '__main__':

    train = pd.read_csv('labeledTrainData.tsv', delimiter='\t', quoting=3, header=0)
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["review"].size
    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []        
    
    for i in range(num_reviews):
        if( (i+1)%1000 == 0 ):
            print ("Review %d of %d\n" % ( i+1, num_reviews ))
        clean_train_reviews.append(review_to_words(train["review"][i]))
    
    vectorizer = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None,stop_words=None,max_features=500)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    print(train_data_features.shape)
    # see all vocabs 
    vocab = vectorizer.get_feature_names()
    print(vocab)

    

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)
    # For each, print the vocabulary word and the number of times it 
    # appears in the training set
    for tag, count in zip(vocab, dist):
        print("%s %s", count, tag)

    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100) 

    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( train_data_features, train["sentiment"] )

    # Read the test data
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3 )

    # Verify that there are 25,000 rows and 2 columns
    print(test.shape)

    # Create an empty list and append the clean reviews one by one
    num_reviews = len(test["review"])
    clean_test_reviews = [] 

    print( "Cleaning and parsing the test set movie reviews...\n")
    for i in range(num_reviews):
        if( (i+1) % 1000 == 0 ):
            print("Review %d of %d\n" % (i+1, num_reviews))
        clean_review = review_to_words( test["review"][i] )
        clean_test_reviews.append( clean_review )

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

    # Use pandas to write the comma-separated output file
    output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
    print('Done!!!')