import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    device = 0 if torch.cuda.is_available() else -1
    return pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=device)

# Preprocess text
def preprocess_text(text):
    if isinstance(text, str):  # Check if the input is a string
        text = text.lower()
        text = re.sub(r'[^a-z A-Z\s]', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    return ''  # Return an empty string for non-string inputs

# Perform sentiment analysis using BERT in batches
def analyze_sentiment(texts, sentiment_pipeline):
    sentiments = sentiment_pipeline(texts)
    sentiment_mapping = {
        '1 star': 'Very Negative',
        '2 stars': 'Negative',
        '3 stars': 'Neutral',
        '4 stars': 'Positive',
        '5 stars': 'Very Positive'
    }
    sentiment_labels = [sentiment_mapping.get(sent['label'], 'Unknown') for sent in sentiments]
    return sentiment_labels  # Return only labels

# Generate and display word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap='plasma',  # Changed colormap for better aesthetics
                          max_words=200,  # Limit the number of words
                          contour_color='steelblue',  # Add contour color
                          contour_width=1).generate(text)  # Generate word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def main():
    st.set_page_config(page_title="Sentiment360", page_icon="📊", layout="wide")

    # Sidebar
    st.sidebar.title("Sentiment360")
    st.sidebar.subheader("Customer Review Analysis Platform")
    st.sidebar.markdown("---")
    st.sidebar.header("Features")
    st.sidebar.markdown("""
    - 📊 **Sentiment Analysis**: Analyze customer sentiments from reviews.
    - ☁️ **Word Cloud Generation**: Visualize the most common words in reviews.
    """)

    # Main content
    st.title("Sentiment360 Analysis")
    st.markdown("Welcome to the **Sentiment360** application! This tool allows you to analyze customer reviews and visualize sentiments effectively.")
   
    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display original dataframe
            st.subheader("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Column selection
            text_column = st.selectbox(
                "Select the column containing reviews:",
                df.columns
            )
            
            if st.button("Analyze Sentiment"):
                # Create progress bar
                progress_bar = st.progress(0)
                
                # Load the sentiment analysis pipeline
                sentiment_pipeline = load_bert_model()
                
                # Perform sentiment analysis
                labels = []
                total_rows = len(df)
                
                for idx, text in enumerate(df[text_column]):
                    label = analyze_sentiment([text], sentiment_pipeline)[0]  # Process one row at a time
                    labels.append(label)
                    progress_bar.progress((idx + 1) / total_rows)
                
                # Add results to dataframe
                df['Sentiment Label'] = labels
                
                # Display results
                st.subheader("Analysis Results:")
                st.dataframe(df)
                
                # Create visualization
                sentiment_counts = df['Sentiment Label'].value_counts()
                fig = px.pie(values=sentiment_counts.values, 
                           names=sentiment_counts.index,
                           title='Sentiment Distribution')
                st.plotly_chart(fig)
                
                # Generate and display word cloud
                all_reviews = ' '.join(df[text_column].dropna().astype(str))
                wordcloud_fig = generate_wordcloud(all_reviews)
                st.subheader("Word Cloud of Reviews")
                st.pyplot(wordcloud_fig)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

