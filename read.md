<!-- import pandas as pd   #used for data manipulation and analysis.
import re             #regular expression (regex) library, which is used for string matching and manipulation

import nltk
from nltk.corpus import stopwords

# These import the Natural Language Toolkit (nltk) library and the stopwords module from it.
# The stopwords module contains a list of common words (like "the", "is", etc.) 
# that are often removed from text during preprocessing 

#TfidfVectorizer is used to convert a collection of raw documents to a matrix of TF-IDF features.
#train_test_split is used to split arrays or matrices into random train and test subsets.
#LogisticRegression is used to implement logistic regression for binary classification.
#accuracy_score and classification_report are used to evaluate the performance of the classification model.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#These lines download the stopwords data and create a set of English stopwords.

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset.........................
#These lines load a CSV file into a pandas DataFrame

df = pd.read_csv('C:\\Users\\abdul\\Desktop\\NLP\\Effective\\Research_data-set\\covid2021.csv', encoding='latin-1', header=None)
df.columns = ['query', 'url', 'title', 'upload_date', 'channel', 'views', 'likes', 'dislikes', 'comment_count', 'comment_text', 'comment_author', 'comment_date', 'comment_likes', 'DATE'] + [f'Unnamed: {i}' for i in range(14, 66)]

# Preprocess the text........................
#Removing URLs, mentions, and hashtags.
#Removing non-word characters.
#Replacing multiple spaces with a single space.
#Converting the text to lowercase.
#Removing stopwords.

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+|@\S+|#\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

#This applies the preprocess_text function to the text column
#  of the DataFrame and stores the result in a new column called Processed_Text
df['Processed_Text'] = df['text'].apply(preprocess_text)

# Vectorize text
#These lines vectorize the processed text using TF-IDF. max_features=5000 limits
#  the number of features to 5000. The y variable is created by converting
#  the target column to binary values: 1 if the value is 4, otherwise 0.
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Processed_Text']).toarray()
y = df['target'].apply(lambda x: 1 if x == 4 else 0)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n {classification_report(y_test, y_pred)}")

# Predict sentiment
#This function predicts the sentiment of a given text by preprocessing it, 
# vectorizing it, and then using the trained model to make a prediction.
#  It returns "Positive" if the sentiment is positive (1), otherwise "Negative".
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    sentiment = model.predict(vectorized_text)
    return "Positive" if sentiment[0] == 1 else "Negative"

#These lines test the predict_sentiment function
#  with two sample texts and print the results.
print(predict_sentiment("I love this product!"))
print(predict_sentiment("I hate this service.")) -->
