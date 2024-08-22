# Assessment 3 - Webcrawler and NLP System
# James Ross (14472266)
# PART 2 - Data Preprocessing, Exploratory Analysis, and NLP Models
# For project on GitHub: https://github.com/jimy-r/MA5851_Assessment3

import json
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from bertopic import BERTopic
from sklearn.decomposition import PCA

# Download required NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the JSON data
with open('raw_data.json', 'r') as f:
    data = json.load(f)

# Extract relevant fields
articles = []
for item in data:
    article = {
        'title': item.get('title', ''),
        'description': item.get('description', ''),
    }
    articles.append(article)

# Convert to DataFrame
df = pd.DataFrame(articles)

# Text Cleaning
def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

df['cleaned_title'] = df['title'].apply(clean_text)
df['cleaned_description'] = df['description'].apply(clean_text)

# Summary Statistics
print("Summary Statistics:")
print(df.describe(include='all'))

# Text Length Analysis
df['title_length'] = df['cleaned_title'].apply(lambda x: len(x.split()))
df['description_length'] = df['cleaned_description'].apply(lambda x: len(x.split()))

print("\nText Length Analysis:")
print(df[['title_length', 'description_length']].describe())

# Visualize Text Length Distributions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['title_length'], bins=30, kde=True)
plt.title('Title Length Distribution')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df['description_length'], bins=30, kde=True)
plt.title('Description Length Distribution')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Save cleaned data to CSV for review
df[['cleaned_title', 'cleaned_description']].to_csv('cleaned_data_new.csv', index=False)

print("Text cleaning complete. Cleaned data saved to 'cleaned_data_new.csv'.")

# Remove rows with one-word titles and empty descriptions
df_filtered = df[(df['cleaned_title'].apply(lambda x: len(x.split())) > 1) & (df['cleaned_description'].apply(lambda x: len(x.strip()) > 0))].copy()

# Save the filtered data to a new CSV file
df_filtered[['cleaned_title', 'cleaned_description']].to_csv('cleaned_updated.csv', index=False)

print("Filtered data saved to 'cleaned_updated.csv'. Summary statistics after filtering:")

# Summary Statistics after filtering
print(df_filtered.describe(include='all'))

# Text Length Analysis after filtering
print("\nText Length Analysis after filtering:")
print(df_filtered[['title_length', 'description_length']].describe())

# Proceed with Tokenization
df_filtered['tokenized_title'] = df_filtered['cleaned_title'].apply(word_tokenize)
df_filtered['tokenized_description'] = df_filtered['cleaned_description'].apply(word_tokenize)

# Word Count Distribution (Before Stopword Removal)
all_words = df_filtered['tokenized_title'].explode().tolist() + df_filtered['tokenized_description'].explode().tolist()
word_freq = Counter(all_words)

# Convert to DataFrame for analysis
word_freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'count'])

# Top 20 most common words (Before Stopword Removal)
top_words = word_freq_df.nlargest(20, 'count')

# Visualize the Top 20 Words (Before Stopword Removal)
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='word', data=top_words)
plt.title('Top 20 Most Common Words (Before Stopword Removal)')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()

# Visualize the distribution of word frequencies (Before Stopword Removal)
plt.figure(figsize=(12, 6))
sns.histplot(word_freq_df['count'], bins=50, kde=True)
plt.title('Word Frequency Distribution (Before Stopword Removal)')
plt.xlabel('Word Frequency')
plt.ylabel('Count of Words')
plt.yscale('log')
plt.show()

# Stopword Removal using NLTK's stopwords list
stop_words = set(stopwords.words('english'))

df_filtered['tokenized_title'] = df_filtered['tokenized_title'].apply(lambda x: [word for word in x if word not in stop_words])
df_filtered['tokenized_description'] = df_filtered['tokenized_description'].apply(lambda x: [word for word in x if word not in stop_words])

# Word Count Distribution (After Stopword Removal)
all_words = df_filtered['tokenized_title'].explode().tolist() + df_filtered['tokenized_description'].explode().tolist()
word_freq = Counter(all_words)

# Convert to DataFrame for analysis
word_freq_df = pd.DataFrame(word_freq.items(), columns=['word', 'count'])

# Top 20 most common words (After Stopword Removal)
top_words = word_freq_df.nlargest(20, 'count')

# Visualize the Top 20 Words (After Stopword Removal)
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='word', data=top_words)
plt.title('Top 20 Most Common Words (After Stopword Removal)')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()

# Visualize the distribution of word frequencies (After Stopword Removal)
plt.figure(figsize=(12, 6))
sns.histplot(word_freq_df['count'], bins=50, kde=True)
plt.title('Word Frequency Distribution (After Stopword Removal)')
plt.xlabel('Word Frequency')
plt.ylabel('Count of Words')
plt.yscale('log')
plt.show()

# Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize(tokens):
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized

df_filtered['processed_title'] = df_filtered['tokenized_title'].apply(lemmatize)
df_filtered['processed_description'] = df_filtered['tokenized_description'].apply(lemmatize)

# Save the processed data to a final CSV file
df_filtered[['processed_title', 'processed_description']].to_csv('preprocessed_data.csv', index=False)

print("Data preprocessing complete. Processed titles and descriptions saved to 'preprocessed_data.csv'.")

# Combine titles and descriptions for embedding
df_filtered['combined_text'] = df_filtered['processed_title'].apply(' '.join) + ' ' + df_filtered['processed_description'].apply(' '.join)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

silhouette_scores = []
davies_bouldin_scores = []
fold = 1

for train_index, test_index in kf.split(df_filtered):
    train_data = df_filtered.iloc[train_index]
    test_data = df_filtered.iloc[test_index]
    
    # Generate Embeddings with BERTopic
    bertopic_model = BERTopic()
    train_embeddings = bertopic_model.fit_transform(train_data['combined_text'].tolist())[1]  # Get the embeddings
    test_embeddings = bertopic_model.transform(test_data['combined_text'].tolist())[1]

    # Ensure embeddings are 2D arrays
    if train_embeddings.ndim == 1:
        train_embeddings = np.expand_dims(train_embeddings, axis=1)
    if test_embeddings.ndim == 1:
        test_embeddings = np.expand_dims(test_embeddings, axis=1)

    # Cluster Embeddings with KMeans
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(train_embeddings)
    
    # Get cluster labels for both train and test data
    train_labels = kmeans.labels_
    test_labels = kmeans.predict(test_embeddings)

    # Evaluate the Model
    silhouette_avg = silhouette_score(train_embeddings, train_labels)
    davies_bouldin_avg = davies_bouldin_score(train_embeddings, train_labels)
    
    silhouette_scores.append(silhouette_avg)
    davies_bouldin_scores.append(davies_bouldin_avg)
    
    print(f"Fold {fold} - Silhouette Score: {silhouette_avg}, Davies-Bouldin Index: {davies_bouldin_avg}")
    fold += 1

# Average Scores
print(f"Average Silhouette Score across all folds: {np.mean(silhouette_scores)}")
print(f"Average Davies-Bouldin Index across all folds: {np.mean(davies_bouldin_scores)}")

# Step 4: Express Cluster Topics in Words
def get_top_words_for_clusters(df, cluster_labels, top_n=10):
    df['cluster'] = cluster_labels
    top_words_per_cluster = {}
    
    for cluster_num in np.unique(cluster_labels):
        cluster_data = df[df['cluster'] == cluster_num]
        all_words = cluster_data['combined_text'].str.split().explode()
        word_freq = Counter(all_words)
        top_words_per_cluster[cluster_num] = dict(word_freq.most_common(top_n))
    
    return top_words_per_cluster

# Use the test data of the last fold for topic extraction
last_fold_test_data = df_filtered.iloc[test_index]  # Get the test data from the last fold
last_fold_test_labels = test_labels

top_words_per_cluster = get_top_words_for_clusters(last_fold_test_data, last_fold_test_labels)

# Print and plot top words for each cluster
for cluster_num, words in top_words_per_cluster.items():
    print(f"Cluster {cluster_num}:")
    print(words)
    print()

# Plot top words for each cluster
for cluster_num, words in top_words_per_cluster.items():
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(words.values()), y=list(words.keys()))
    plt.title(f'Top Words for Cluster {cluster_num}')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.show()

# Apply BERTopic for Topic Modeling
bertopic_model_only = BERTopic()
topics, probabilities = bertopic_model_only.fit_transform(df_filtered['combined_text'].tolist())

# Evaluate the BERTopic model using silhouette and Davies-Bouldin scores

# Extract the embeddings from the BERTopic model
bertopic_embeddings_only = bertopic_model_only.transform(df_filtered['combined_text'].tolist())[1]

# Ensure the embeddings are 2D arrays
if bertopic_embeddings_only.ndim == 1:
    bertopic_embeddings_only = np.expand_dims(bertopic_embeddings_only, axis=1)

# Use KMeans labels as the clustering criterion
kmeans_bertopic_only = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_bertopic_only.fit(bertopic_embeddings_only)
bertopic_labels_only = kmeans_bertopic_only.labels_

# Calculate Silhouette Score
silhouette_score_bertopic_only = silhouette_score(bertopic_embeddings_only, bertopic_labels_only)

# Calculate Davies-Bouldin Index
davies_bouldin_score_bertopic_only = davies_bouldin_score(bertopic_embeddings_only, bertopic_labels_only)

print(f"BERTopic Only - Silhouette Score: {silhouette_score_bertopic_only}")
print(f"BERTopic Only - Davies-Bouldin Index: {davies_bouldin_score_bertopic_only}")

# Compare scores with the BERTopic + KMeans model
average_silhouette_kmeans = np.mean(silhouette_scores)
average_davies_bouldin_kmeans = np.mean(davies_bouldin_scores)

print(f"Comparison of Models:")
print(f"Average Silhouette Score for BERTopic + KMeans: {average_silhouette_kmeans}")
print(f"Silhouette Score for BERTopic Only: {silhouette_score_bertopic_only}")

print(f"Average Davies-Bouldin Index for BERTopic + KMeans: {average_davies_bouldin_kmeans}")
print(f"Davies-Bouldin Index for BERTopic Only: {davies_bouldin_score_bertopic_only}")

# Extract and display the top words for each topic in the BERTopic-only model
# Get the topics and their top words
top_words_per_topic_bertopic_only = bertopic_model_only.get_topics()

# Print and plot top words for each topic
for topic_num, words in top_words_per_topic_bertopic_only.items():
    if topic_num == -1:
        continue  # Skip the outlier topic (-1)
    
    print(f"Topic {topic_num}:")
    print(dict(words))
    print()

    # Plot the top words for each topic
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[freq for _, freq in words], y=[word for word, _ in words])
    plt.title(f'Top Words for Topic {topic_num} (BERTopic Only)')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.show()