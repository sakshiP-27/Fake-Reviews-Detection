import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns  

# Load the dataset (replace 'your_labeled_dataset.csv' with the actual file path)
df = pd.read_csv('yelp.csv')

# Take 50% of the dataset
df_sampled = df.sample(frac=0.5, random_state=42)

# Step 1: Filter dataset based on constraints
df_filtered = df_sampled[(df_sampled['Rating'].isin([4, 5])) & (df_sampled['Date'].str.contains('2014'))]

# Step 2: Text preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Create a copy of the DataFrame to avoid SettingWithCopyWarning
df_filtered = df_filtered.copy()
df_filtered['Review'] = df_filtered['Review'].apply(preprocess_text)

# Drop 'User_id' and 'Product_id' columns
df_filtered.drop(columns=['User_id', 'Product_id'], inplace=True)

# Step 3: Calculate sentiments for each review
def calculate_sentiment(text):
    analysis = TextBlob(text)
    # Assign sentiment labels based on polarity (adjust threshold as needed)
    return 'Positive' if analysis.sentiment.polarity > 0 else 'Negative'

# Create another copy to avoid SettingWithCopyWarning
df_filtered = df_filtered.copy()
# Apply sentiment analysis to the 'Review' column
df_filtered['Sentiment'] = df_filtered['Review'].apply(calculate_sentiment)

# Filter out positive reviews
df_filtered = df_filtered[df_filtered['Sentiment'] == 'Positive']
df_filtered.drop(columns=['Sentiment'], inplace=True)  # Drop the temporary sentiment column

# Step 4: Convert text data to numerical format using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust the number of features as needed
X = vectorizer.fit_transform(df_filtered['Review']).toarray()

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df_filtered['Label'], test_size=0.2, random_state=42)

# Step 6: Standardize the data (for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Data Visualization Steps


# Pie Chart : To visualize the ratings distribution of each real and fake reviews
# For Real Reviews
df_positive_label = df[df["Label"] == 1]

# Count the frequency of each rating having real label
rating_counts = df_positive_label["Rating"].value_counts()

# Create a pie chart
plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Frequency of Ratings for Label = 1")
plt.axis('equal')
plt.show()

# For Fake Reviews
df_negative_label = df[df["Label"] == -1]

# Count the frequency of each rating having fake label
rating_counts = df_negative_label["Rating"].value_counts()

# Create a pie chart
plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Frequency of Ratings for Label = -1")
plt.axis('equal')
plt.show()


# Boxplot : To visualize the length of reviews and compare them as real and fake reviews
df['Review_Length'] = df['Review'].apply(len)

# Plotting the data points
plt.figure(figsize=(8, 6))
sns.boxplot(x='Label', y='Review_Length', data=df, hue='Label', palette='pastel', dodge=True)
plt.title('Review Length Distribution by Label')
plt.xlabel('Label (1: Real, -1: Fake)')
plt.ylabel('Review Length')
plt.legend(title=None)
plt.show()


# Line Graph / Time Series Graph : To visualize the number of reviews given over the time both real and fake (Date Vs No of Reviews)
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

# Plotting the datapoints
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='count', hue='Label', data=df.groupby(['Year', 'Label']).size().reset_index(name='count'), marker='o')
plt.title('Temporal Trends in Number of Reviews by Label')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.show()


# Bar Graph : To visualize the 10 most frequent words present in all fake reviews and giving their frequency
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['Review'])

# Create a DataFrame with word frequencies
word_freq_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Add the 'Label' column to the DataFrame
word_freq_df['Label'] = df['Label']

# Calculate the average frequency of each word for real and fake reviews
average_word_freq = word_freq_df.groupby('Label').mean().transpose()

# Choose the top N words
top_words = average_word_freq.sort_values(by=-1).head(10)  # Replace 10 with the desired number of words

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(data=top_words.reset_index(), x='index', y=-1)
plt.title('Top Words in Fake Reviews')
plt.xlabel('Words')
plt.ylabel('Average Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()


# Wordcloud
from wordcloud import WordCloud
reviews_text = ' '.join(df['Review'].astype(str))

# Generate Word Cloud
wordcloud = WordCloud(width=800, height=400, random_state=42, background_color='white').generate(reviews_text)

# Plotting
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Reviews')
plt.show()


# Heatmap (Comparing the relations among User_Id, Product_Id, Rating, Label)
# Calculate the correlation among columns matrix
corr_matrix = df[['User_id', 'Product_id', 'Rating', 'Label']].corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Heatmap: Correlation between User_Id, Product_Id, Rating, and Label')
plt.show()


# Bar Chart (Applying sentiment analysis and comparing the real and fake reviews)
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis and classify each review
df['Sentiment'] = df['Review'].apply(lambda x: 'Positive' if sia.polarity_scores(x)['compound'] >= 0.5 else ('Negative' if sia.polarity_scores(x)['compound'] <= -0.5 else 'Neutral'))

# Create a grouped bar chart
plt.figure(figsize=(10, 8))
sns.countplot(x='Sentiment', hue='Label', data=df, palette={1: 'blue', -1: 'red'})
plt.title('Sentiment Analysis: Polarized Sentiment Distribution for Real and Fake Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# Step 7: Build and evaluate SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test_scaled)

# Evaluate SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

print("SVM Model:")
print(f"Accuracy: {accuracy_svm:.4f}")
print(f"F1 Score: {f1_svm:.4f}")

# Step 8: Build and evaluate Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\nRandom Forest Model:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")

# User Interface
def predict_review(model, vectorizer, scaler=None):
    # Take user input
    user_review = input("Enter your review: ")

    # Preprocess the user input
    user_review = preprocess_text(user_review)

    # Convert the user input to numerical format using TF-IDF
    user_review_vectorized = vectorizer.transform([user_review]).toarray()

    # Standardize the data if using SVM
    if scaler is not None:
        user_review_vectorized = scaler.transform(user_review_vectorized)

    # Make prediction using the selected model
    prediction = model.predict(user_review_vectorized)[0]

    # Display the prediction
    if prediction == 1:
        print("Prediction: Real Review")
    else:
        print("Prediction: Fake Review")

# Choose the model
model_choice = input("Choose the model (SVM or Random Forest): ").lower()

if model_choice == 'svm':
    selected_model = svm_model
    selected_vectorizer = vectorizer
    selected_scaler = scaler
elif model_choice == 'random forest':
    selected_model = rf_model
    selected_vectorizer = vectorizer
    selected_scaler = None
else:
    print("Invalid model choice. Please choose either 'SVM' or 'Random Forest'.")
    exit()

# Call the function to predict a user input review
predict_review(selected_model, selected_vectorizer, selected_scaler)
