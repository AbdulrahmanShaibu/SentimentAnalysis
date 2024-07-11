import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Download and set stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv('C:\\Users\\abdul\\Desktop\\NLP\\Effective\\Research_data-set\\covid2021.csv', encoding='latin-1', header=None, low_memory=False)
df.columns = ['query', 'url', 'title', 'upload_date', 'channel', 'views', 'likes', 'dislikes', 'comment_count', 'comment_text', 'comment_author', 'comment_date', 'comment_likes', 'DATE'] + [f'Unnamed: {i}' for i in range(14, 66)]

print(df.columns)

# Preprocess the text
def preprocess_text(text, file):
    file.write("Original Text: " + text + "\n")
    
    text = re.sub(r'http\S+|www\S+|https\S+|@\S+|#\S+', '', text, flags=re.MULTILINE)
    file.write("Removed URLs, Mentions, Hashtags: " + text + "\n")
    
    text = re.sub(r'\W', ' ', text)
    file.write("Removed Non-word Characters: " + text + "\n")
    
    text = re.sub(r'\s+', ' ', text)
    file.write("Replaced Multiple Spaces with Single Space: " + text + "\n")
    
    text = text.lower()
    file.write("Converted to Lowercase: " + text + "\n")
    
    text = ' '.join([word for word in text.split() if word not in stop_words])
    file.write("Removed Stopwords: " + text + "\n\n")
    
    return text

# Specify document path on the desktop
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
file_path = os.path.join(desktop_path, 'preprocessing_steps.txt')

# Open a file to write preprocessing steps
with open(file_path, 'w', encoding='utf-8') as file:
    # Apply preprocessing to the correct column
    df['Processed_Text'] = df['comment_text'].apply(lambda x: preprocess_text(str(x), file))

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Processed_Text']).toarray()

# Create target variable based on likes and dislikes
df['target'] = df.apply(lambda row: 1 if row['likes'] > row['dislikes'] else 0, axis=1)

# Visualize data distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df)
plt.title('Distribution of Sentiment Labels')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, df['target'], test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n {classification_report(y_test, y_pred)}")

# Visualize model performance
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).iloc[:-1, :].T, annot=True, cmap='coolwarm')
plt.title('Classification Report')
plt.show()

# Predict sentiment
positive_count = sum(y_test == 1)
negative_count = sum(y_test == 0)

# Visualize number of positives and negatives
plt.figure(figsize=(8, 6))
sns.barplot(x=['Positive', 'Negative'], y=[positive_count, negative_count], palette='viridis')
plt.title('Number of Positive and Negative Comments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
