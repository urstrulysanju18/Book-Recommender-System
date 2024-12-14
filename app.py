import pickle
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the required data
model = pickle.load(open('model/KNN_Model.pkl', 'rb'))
books = pickle.load(open('model/book_names.pkl', 'rb'))
final_ratings = pickle.load(open('model/final_ratings.pkl', 'rb'))
df_pivot = pickle.load(open('model/df_pivot.pkl', 'rb'))
popular_df = pickle.load(open('model/popular_df.pkl', 'rb'))

# Function to fetch poster URLs
def fetch_poster(suggestion):
    poster_url = []
    authors = []
    publishers = []
    ratings = []

    for book_id in suggestion[0]:
        book_name = df_pivot.index[book_id]
        df_final = final_ratings[final_ratings['title'] == book_name].drop_duplicates('title')

        url = df_final.iloc[0]['image_url']
        poster_url.append(url)
        author = df_final.iloc[0]['author']
        authors.append(author)
        rating = df_final.iloc[0]['rating']
        ratings.append(rating)
        publisher = df_final.iloc[0]['publisher']
        publishers.append(publisher)

    return poster_url, authors, publishers, ratings

# Function to recommend books using KNN
def recommend_book_knn(book_name):
    books_list = []
    book_id = np.where(df_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(df_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url, authors, publishers, ratings = fetch_poster(suggestion)
    books_list = [df_pivot.index[suggestion[0][i]] for i in range(len(suggestion[0]))]

    return books_list, poster_url, authors, publishers, ratings

# Function to recommend books using Cosine Similarity
def recommend_book_cosine(book_name):
    book_id = np.where(df_pivot.index == book_name)[0][0]
    cosine_sim = cosine_similarity(df_pivot)
    scores = list(enumerate(cosine_sim[book_id]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:7]

    suggestion = [x[0] for x in sorted_scores]
    suggestion = [np.array(suggestion)]

    poster_url, authors, publishers, ratings = fetch_poster(suggestion)
    books_list = [df_pivot.index[suggestion[0][i]] for i in range(len(suggestion[0]))]

    return books_list, poster_url, authors, publishers, ratings

# Streamlit App
st.header('Book Recommender System Using Machine Learning')

# Sidebar
st.sidebar.title("Menu")
option = st.sidebar.selectbox(
    "Choose a section",
    ["Popular Books", "Recommend Books"]
)

if option == "Popular Books":
    st.subheader("Top 50 Popular Books")
    cols = st.columns(3)
    for idx, col in enumerate(cols):
        with col:
            for i in range(idx, len(popular_df), 3):
                st.image(popular_df['image_url'].values[i], width=150)
                st.write(f"**{popular_df['title'].values[i]}** by {popular_df['author'].values[i]}")
                st.write(f"Votes: {popular_df['num_of_ratings'].values[i]}, Rating: {popular_df['average_rating'].values[i]:.2f}")
                st.write("----")

if option == "Recommend Books":
    st.subheader("Recommend Books")
    
    selected_books = st.selectbox(
        "Type or select a book from the dropdown",
        books
    )

    recommend_method = st.radio(
        "Choose recommendation techniques",
        ("KNN", "Cosine Similarity")
    )

    if st.button('Show Recommendation'):
        if recommend_method == "KNN":
            recommended_books, poster_url, authors, publishers, ratings = recommend_book_knn(selected_books)
        else:
            recommended_books, poster_url, authors, publishers, ratings = recommend_book_cosine(selected_books)

        cols = st.columns(5)
        for row in range(2):
            cols = st.columns(5)
            for idx, col in enumerate(cols, start=1):
                with col:
                    index = row * 5 + idx
                    if index < len(recommended_books):
                        st.image(poster_url[index], width=100)
                        st.write(f"**{recommended_books[index]}**")
                        st.write(f"Author: {authors[index]}")
                        st.write(f"Publisher: {publishers[index]}")
                        st.write(f"Rating: {ratings[index]:.2f}")
