import streamlit as st
import pandas as pd
import numpy as np
import requests


st.set_page_config(layout="wide")

### Title ###
st.title("CS 598 PSL Project 4: Movie Recommender")

# init
@st.cache_data
def load_data():
    popular = pd.read_csv('https://drive.google.com/uc?id=1JTQqNhPD8pqDUXtwOMEffRoKr4OjiDh2', index_col=0)
    top_similarity = pd.read_csv('https://drive.google.com/uc?id=1C1oXdcGSh21f_5rMhKQlHa-GfGY79QSq', index_col=0)
    movie_titles = {}
    response = requests.get('https://liangfgithub.github.io/MovieData/movies.dat?raw=true')
    for line in response.text.split('\n'):
        if line:
            id, title, _ = line.split('::')
            movie_titles['m'+id] = title
    return popular, top_similarity, movie_titles


popular, top_similarity, movie_titles = load_data()

### Step 1 ###
st.header("Step 1: Rate Movies!")

# display movies
ratings = []
num_movies, num_cols, i = 100, 5, 0
movies = popular.index.tolist()
cols = st.columns(num_cols, border=False)
for col in cols:
    with col.container(height=1000):
        for _ in range(num_movies//num_cols):
            with st.container(border=True):
                st.image(f"https://liangfgithub.github.io/MovieImages/{movies[i][1:]}.jpg?raw=true")
                st.text(movie_titles[movies[i]])
                rating = st.feedback("stars", key=f'rating-{i}')
                ratings.append(rating)
                i += 1

### Step 2 ###
st.header("Step 2: Discover Movies!")

def myIBCF():

    # init
    global top_similarity, popular, movie_titles, ratings, recommendations
    n = top_similarity.shape[0]

    # user rating
    newuser = np.full((n,1), np.nan)
    for idx, rating in enumerate(ratings):
        if rating is not None:
            id = popular.index[idx]
            newuser[top_similarity.index.get_loc(id), 0] = rating + 1

    # rating prediction
    prediction = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(newuser[i, 0]):
            w, s = newuser.reshape(n), top_similarity.iloc[i].to_numpy()
            j = ~np.isnan(w) & ~np.isnan(s)
            w, s = w[j], s[j]
            prediction[i] = np.sum(s * w) / np.sum(s)

    idx = np.argsort(np.nan_to_num(prediction, -np.inf))[::-1]

    top_ten = []

    # top predictions
    i = 0
    while len(top_ten) < 10 and i < n and not np.isnan(prediction[idx[i]]):
        id = top_similarity.index[idx[i]]
        top_ten.append(id)
        i += 1

    # most popular
    i = 0
    while len(top_ten) < 10 and i < len(popular):
        id = popular.index[i]
        if id not in top_ten and np.isnan(newuser[top_similarity.index.get_loc(id), 0]):
            top_ten.append(id)
        i += 1

    # display top ten
    with recommendations:
        num_movies, num_rows, num_cols, i = 10, 2, 5, 0
        for _ in range(num_rows):
            cols = st.columns(num_cols)
            for col in cols:
                with col.container():
                    movie = top_ten[i]
                    st.image(f"https://liangfgithub.github.io/MovieImages/{movie[1:]}.jpg?raw=true")
                    st.text(movie_titles[movie])
                    i += 1

    return


st.button('Get Recommendations', on_click=myIBCF)
recommendations = st.container()