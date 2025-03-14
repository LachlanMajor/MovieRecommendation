import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

set_time = time.time()
print(f"Time to set up: {set_time}")

# User ratings from letterboxd csv 
ratings = pd.read_csv("Recommender/ratings.csv")
ratings = ratings.drop(columns=['Date', 'Letterboxd URI'])

ratings_time = time.time()
print(f"Time to load in ratings: {ratings_time - set_time}")


# Collection of movie titles with their genres
movies = pd.read_csv("Recommender/movies.csv")
# Remove (year) from titles
movies[['title', 'year']] = movies['title'].str.extract(r'^(.*)\s\((\d{4})\)$')
movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
movies['year'] = movies['year'].fillna(0).astype(int)
# Remove Nan titles
movies= movies.dropna(subset=['title']).reset_index(drop=True)

movies_time = time.time()
print(f"Time to load in movies: {movies_time - ratings_time}")


#Colletion of ratings to be used in the correlation matrix.
all_ratings = pd.read_csv("Recommender/allratings.csv", index_col=False)
all_ratings = all_ratings.drop(columns=['timestamp'])
all_ratings = all_ratings.dropna(subset=['userId', 'movieId', 'rating'])
# Merge movies into all_ratings to have title, genres, and year
all_ratings = pd.merge(all_ratings, movies)
# Remove anything in brackets in the title for things like foreign films and the original title
all_ratings['title'] = all_ratings['title'].str.replace(r'\s*\(.*?\)', '', regex=True)
load_in = time.time()
print(f"Time to load in data: {load_in - movies_time}")

# Any title with The, A, or An has it moved to the front
def move_article_to_front(title):
    match = re.match(r'^(.*?),\s*(The|A|An)$', title, re.IGNORECASE)
    if match:
        rest_of_title, article = match.groups()
        return f"{article} {rest_of_title}"
    return title

ratings['Name'] = ratings['Name'].apply(move_article_to_front)
#all_ratings['title'] = all_ratings['title'].apply(move_article_to_front)
move_article_to_front_time = time.time()

print(f"Time to move articles to front: {move_article_to_front_time - load_in}")


# Limit the included movies to movies with a minimum number of ratings
ratings_per_movie = all_ratings.groupby('movieId').size()
accepted_movies = ratings_per_movie[ratings_per_movie >= 1000].index
filtered_ratings = all_ratings[all_ratings['movieId'].isin(accepted_movies)]
num_movies = filtered_ratings['movieId'].nunique()
# print(f"Movies: {num_movies}")

num_movies_times = time.time()
print(f"Time to filter movies: {num_movies_times - move_article_to_front_time}")

from scipy.sparse import coo_matrix


movie_similarity_df = None

def calculate_similarity(sparse_matrix):
    # Calculate the cosine similarity between movies. Transpose the matrix to find item-item similarity instead of user-user
    similarity = cosine_similarity(sparse_matrix.T, dense_output=False)
    
    # Calculate the signifance of the correlation based on the number of ratings. i.e more ratings should mean more impact on the correlation calculation
    n_ratings = (sparse_matrix != 0).sum(axis=0)
    min_ratings = 1000
    significance_matrix = n_ratings.T.dot(n_ratings) / (n_ratings.T.dot(n_ratings) + min_ratings)
    
    return similarity.multiply(significance_matrix)

# Make a matrix of all movie correlations
def make_matrix(fitlered_ratings):
    global movie_similarity_df
    if movie_similarity_df is not None:
        return movie_similarity_df
    global_mean = fitlered_ratings['rating'].mean()
    
    movie_means = filtered_ratings.groupby('title')['rating'].mean().to_dict()
    
    user_ids = filtered_ratings['userId'].astype('category').cat.codes
    title_ids = filtered_ratings['title'].astype('category').cat.codes
    rating_values = filtered_ratings['rating']
    
    # Sparse matrix made with coo_matrix to include only non 0 values, therefore saving space and time
    sparse_matrix = coo_matrix((rating_values, (user_ids, title_ids))).tocsr()
    
    movie_similarity = calculate_similarity(sparse_matrix)
    
    titles = filtered_ratings['title'].astype('category').cat.categories
    movie_similarity_df = pd.DataFrame(
        movie_similarity.toarray(),
        index=titles,
        columns=titles
    )
    
    return movie_similarity_df, global_mean, movie_means

movie_similarity_df, global_mean, movie_means = make_matrix(filtered_ratings)

matrix_time = time.time()
print(f"Time to make matrix: {matrix_time - num_movies_times}")

import pandas as pd
from rapidfuzz import process, fuzz
from difflib import ndiff


# Some movie titles differ slightly in the dataset so extremly similar titles are matched to find movies with different titles
def replace_titles(user_df, movie_similarity_df, threshold=95):
    title_map = {}
    
    dataset_titles = movie_similarity_df.columns
    # Iterate over user rated movies
    for i, user_movie in enumerate(user_df['Name']):
        
        # Skip if the movie is already in the dataset
        if user_movie in dataset_titles:
            continue

        # Find the best match in the dataset
        match, score, index = process.extractOne(
            user_movie,
            dataset_titles,
            scorer=fuzz.token_sort_ratio
        )
        
        # If the similarity score isn't high enough, skip
        if score >= threshold:
            # If the match is already in the user rating, it is a different movie so skip it
            if match not in user_df['Name'].values:
                #print(f"Matched '{user_movie}' with '{match}'")
                user_df.at[i, 'Name'] = match

    return user_df

ratings = replace_titles(ratings, movie_similarity_df, threshold=90)

titles_time = time.time()
print(f"Time to replace titles: {titles_time - matrix_time}")

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def find_expected_rating(title, user_ratings, movie_similarity_df, global_mean, movie_means):
    # if the movie isn't in the dataframe it has no correlations and is therefore useless and should be ignored
    if title not in movie_similarity_df.index:
        return None
    
    # Don't include the movie in the similarities
    similarities = movie_similarity_df[title].drop(title)
    
    movie_ratings = user_ratings.set_index('Name')['Rating']
    
    # Similarities between movies the user hasn't seen shouldn't impact the expected score
    valid_similarities = similarities[similarities.index.isin(movie_ratings.index)]
    
    # If they haven't seen anything just return the average rating of the movie
    if len(valid_similarities) == 0:
        return movie_means.get(title, global_mean)
    
    valid_ratings = movie_ratings[valid_similarities.index]
    
    numerator = 0
    denominator = 0
    
    # Calculate for each movie, how far from the average the user's rating is
    for movie, similarity in valid_similarities.items():
        if similarity > 0.2:
            rating_deviation = valid_ratings[movie] - movie_means.get(movie, global_mean)
            # Multiply with similarity so the more similar movies have a greater impact on the overall expected deviation
            # Use squared similarity so higher similarity has even greater impact
            numerator += (similarity**2) * rating_deviation
            denominator += abs(similarity**2)
    
    if denominator == 0:
        return movie_means.get(title, global_mean)
    
    # Based on the users weighted deviation above, predict the users rating for the movie
    prediction = movie_means.get(title, global_mean) + (numerator / denominator)
    
    if isinstance(prediction, pd.Series):
        prediction = prediction.iloc[0]
    return max(0.0, min(5.0, prediction))

predict_time = time.time()
print(f"Time to predict: {predict_time - titles_time}")

# Compare each actual rating with the expected rating to evaluate the accuracy of the model
def evaluate_predictions(train_ratings, test_ratings, movie_similarity_df, global_mean, movie_means):
    predictions = []
    actuals = []
    titles = []
    outliers = 0
    Nones = 0
    
    for index, row in test_ratings.iterrows():
        title = row['Name']
        actual_rating = row['Rating']
        
        expected_rating = find_expected_rating(title, train_ratings, movie_similarity_df, global_mean, movie_means)
        
        if expected_rating is not None:
            predictions.append(expected_rating)
            actuals.append(actual_rating)
            titles.append(title)

            # Number of predictions that are drastically different from the actual rating
            if abs(expected_rating - actual_rating) > 1.5:
                outliers += 1
        else:
            Nones += 1
    
    mae = mean_absolute_error(actuals, predictions) if predictions else None
    rmse = np.sqrt(mean_squared_error(actuals, predictions)) if predictions else None
    outlier_percent = (outliers / len(test_ratings)) * 100
    
    return predictions, actuals, titles, mae, rmse, outlier_percent, Nones

train_ratings, test_ratings = train_test_split(ratings, test_size=0.20, random_state=1)
predictions, actuals, titles, mae, rmse, outlier_percent, Nones = evaluate_predictions(train_ratings, test_ratings, movie_similarity_df, global_mean, movie_means)

# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
# print(f"Outlier Percent: {outlier_percent:.2f}%")
# print(f"Number of Nones: {Nones}")
# print(f"Number of test ratings: {len(test_ratings)}")
# print(f"Number of train ratings: {len(train_ratings)}")
# print(f"Number of total ratings: {len(ratings)}")

evaluate_time = time.time()
print(f"Time to evaluate: {evaluate_time - predict_time}")

def recommend_movies(user_ratings, movie_similarity_df, selected_genres, startYear, endYear, anySelected):
    recommendations = []
    ratings_count = all_ratings['title'].value_counts()
    already_rated = set(ratings['Name'])


    # For every movie in the similarity matrix, calculate the expected rating for the user if they haven't seen it
    for movie in movie_similarity_df:
        if movie not in already_rated:
            expected_rating = find_expected_rating(movie, user_ratings, movie_similarity_df, global_mean, movie_means)
            if expected_rating is not None:
                num_ratings = ratings_count.get(movie, 0)

                recommendations.append((movie, expected_rating, num_ratings))

    rec_time = time.time()
    print(f"Time to recommend: {rec_time - evaluate_time}")

    # Sort the recommendations by expected rating and then by popularity
    recommendations.sort(key=lambda x: (x[1], x[2]), reverse=True)
    recommendations = pd.DataFrame(recommendations, columns=['Movie', 'Expected Rating', 'Popularity'])
    recommendations = recommendations.merge(movies[['title', 'genres', 'year']], left_on='Movie', right_on='title', how='left')
    recommendations = recommendations.drop(columns=['title'])
    recommendations = recommendations[recommendations['year'].between(startYear, endYear)]


    recommendations['genres'] = recommendations['genres'].fillna('Any') 
    if "Any" not in selected_genres:
        if anySelected:
            recommendations = recommendations[recommendations['genres'].apply(lambda g: any(genre in selected_genres for genre in g.split('|')))]
        else:
            recommendations = recommendations[recommendations['genres'].apply(lambda g: set(selected_genres).issubset(set(g.split('|'))))]

    rec_time2 = time.time()
    print(f"Time to filter: {rec_time2 - rec_time}")

    total_time = time.time()
    print(f"Total time: {total_time - set_time}")
    return recommendations
    
