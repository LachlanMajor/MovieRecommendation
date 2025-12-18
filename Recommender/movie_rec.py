import pandas as pd
import numpy as np
import os
import sys
from scipy.sparse import coo_matrix, load_npz, save_npz
from sklearn.neighbors import NearestNeighbors
import time
from sklearn.preprocessing import normalize
import json

def setup_data():
    # User ratings from letterboxd csv 
    ratings = pd.read_csv("../../Recommender/ratings.csv", usecols=['Name', 'Rating', 'Year'])

    # Collection of movie titles with their genres
    movies = pd.read_csv("../../Recommender/movies.csv")
    # Remove (year) from titles
    movies[['title', 'year']] = movies['title'].str.extract(r'^(.*)\s\((\d{4})\)$')
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    movies['year'] = movies['year'].fillna(0).astype(int)
    # Remove Nan titles
    movies = movies.dropna(subset=['title']).reset_index(drop=True)
    movies = movies.drop_duplicates(subset=['title', 'year']).reset_index(drop=True)
    
    movies['title_clean'] = movies['title'].str.replace(r'\s*\(.*\)\s*$', '', regex=True)
    mask = movies['title_clean'].str.endswith(', The')
    movies.loc[mask, 'title_clean'] = 'The ' + movies.loc[mask, 'title_clean'].str[:-5]
    movies['title'] = movies['title_clean']
    movies = movies.drop(columns=['title_clean'])

    links = pd.read_csv("../../Recommender/links.csv", usecols=['movieId', 'tmdbId'])
    movies = movies.merge(links, on='movieId', how='left')

    # Colletion of ratings to be used in the correlation matrix.
    all_ratings = pd.read_csv("../../Recommender/allratings.csv")
    all_ratings = all_ratings.drop(columns=['timestamp'])
    all_ratings = all_ratings.dropna(subset=['userId', 'movieId', 'rating'])
    # Merge movies into all_ratings to have title, genres, and year
    all_ratings = pd.merge(all_ratings, movies)
    return ratings, all_ratings

def popular_movies(all_ratings, count):
    # Limit the included movies to movies with a minimum number of ratings
    ratings_per_movie = all_ratings.groupby('movieId').size()
    accepted_movies = ratings_per_movie[ratings_per_movie >= count].index
    filtered_ratings = all_ratings[all_ratings['movieId'].isin(accepted_movies)]
    num_movies = filtered_ratings['movieId'].nunique()
    return filtered_ratings

def kMostSimilar(sparse_matrix, k=50):
    if os.path.exists('../../Recommender/kmost_cache.npz'):
        data = np.load('../../Recommender/kmost_cache.npz')
        return data['inds'], data['sims']
    
    norm_mat = normalize(sparse_matrix.T, norm='l2', axis=1)
    model = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='ball_tree', n_jobs=-1)
    model.fit(norm_mat)
    dists, inds = model.kneighbors(norm_mat)
    sims = 1 - (dists**2 / 2)
    np.savez('../../Recommender/kmost_cache.npz', inds=inds, sims=sims)
    return inds, sims

def map_user_ratings_to_movieids(user_df, filtered_ratings):
    # Create a lookup of title+year to movieId for exact matches
    title_year_to_id = filtered_ratings.groupby(['title', 'year'])['movieId'].first().to_dict()
    
    mapped_ratings = []
    not_found = []
    
    for _, row in user_df.iterrows():
        user_title = row['Name']
        user_year = row.get('Year', None)
        
        # Try exact match first with title+year
        if user_year and (user_title, user_year) in title_year_to_id:
            movie_id = title_year_to_id[(user_title, user_year)]
            mapped_ratings.append({
                'movieId': movie_id,
                'Rating': row['Rating'],
                'original_title': user_title
            })
            continue
        
        else:
            not_found.append((user_title, user_year))

    mapped_df = pd.DataFrame(mapped_ratings)
    
    return mapped_df, not_found

def find_expected_rating(movie_id, user_ratings, sparse_matrix, topk_inds, topk_sims, movie_means, global_mean, movie_id_to_index, threshold=0.2):
    # if the movie isn't in the dataframe it has no correlations and is therefore useless and should be ignored
    if movie_id not in movie_id_to_index:
        return None
    
    user_mean = user_ratings['Rating'].mean()
    user_bias = user_mean - global_mean

    target_ind = movie_id_to_index[movie_id]
    neigh_inds = topk_inds[target_ind]
    neigh_sims = topk_sims[target_ind]

    valid = [movie_index_to_id[idx] in user_ratings['movieId'].values for idx in neigh_inds]
    if sum(valid) == 0:
        return movie_means.get(movie_id, global_mean) + user_bias
    
    valid_inds = neigh_inds[valid]
    valid_sims = neigh_sims[valid]
    valid_movie_ids = [movie_index_to_id[ind] for ind in valid_inds]

    valid_sims = np.array(valid_sims)
    valid_movie_ids = np.array(valid_movie_ids)
    mask_thresh = valid_sims > threshold
    if mask_thresh.sum() == 0:
        return movie_means.get(movie_id, global_mean) + user_bias
    
    valid_sims = valid_sims[mask_thresh]
    valid_movie_ids = valid_movie_ids[mask_thresh]

    user_ratings_dict = dict(zip(user_ratings['movieId'], user_ratings['Rating']))
    numerator = sum((sim**2) * (user_ratings_dict[m] - movie_means.get(m, global_mean)) for sim, m in zip(valid_sims, valid_movie_ids))
    denominator = sum(sim**2 for sim in valid_sims)
    prediction = movie_means.get(movie_id, global_mean) + numerator/denominator + user_bias
    return max(0.0, min(5.0, prediction))

def recommend_movies(user_ratings, all_ratings, sparse_matrix, genres, startYear, endYear, topk_inds, topk_sims, movie_means, global_mean, movie_index_to_id):
    ratings_count = all_ratings.groupby('movieId').size()
    already_rated = set(user_ratings['movieId'])
    
    # Pre-compute user statistics
    user_mean = user_ratings['Rating'].mean()
    user_bias = user_mean - global_mean
    user_ratings_dict = dict(zip(user_ratings['movieId'], user_ratings['Rating']))
    
    # Get all candidate movies
    all_movie_ids = list(movie_index_to_id.values())
    candidate_ids = [mid for mid in all_movie_ids if mid not in already_rated]
    
    # Vectorized prediction
    predictions = []
    for movie_id in candidate_ids:
        if movie_id not in movie_id_to_index:
            continue
            
        target_ind = movie_id_to_index[movie_id]
        neigh_inds = topk_inds[target_ind]
        neigh_sims = topk_sims[target_ind]
        
        # Find which neighbors the user has rated
        neighbor_movie_ids = [movie_index_to_id[idx] for idx in neigh_inds]
        rated_mask = np.array([mid in user_ratings_dict for mid in neighbor_movie_ids])
        
        if not rated_mask.any():
            pred = movie_means.get(movie_id, global_mean) + user_bias
        else:
            valid_sims = neigh_sims[rated_mask]
            valid_movie_ids = np.array(neighbor_movie_ids)[rated_mask]
            
            # Apply threshold
            thresh_mask = valid_sims > 0.2
            if not thresh_mask.any():
                pred = movie_means.get(movie_id, global_mean) + user_bias
            else:
                valid_sims = valid_sims[thresh_mask]
                valid_movie_ids = valid_movie_ids[thresh_mask]
                
                # Vectorized calculation
                user_ratings_arr = np.array([user_ratings_dict[m] for m in valid_movie_ids])
                movie_means_arr = np.array([movie_means.get(m, global_mean) for m in valid_movie_ids])
                
                numerator = np.sum((valid_sims**2) * (user_ratings_arr - movie_means_arr))
                denominator = np.sum(valid_sims**2)
                pred = movie_means.get(movie_id, global_mean) + numerator/denominator + user_bias
                pred = max(0.0, min(5.0, pred))
        
        predictions.append((movie_id, pred, ratings_count.get(movie_id, 0)))
    
    # Sort and create DataFrame
    predictions.sort(key=lambda x: (x[1], x[2]), reverse=True)
    recommendations = pd.DataFrame(predictions, columns=['movieId', 'Rating', 'Popularity'])
    
    recommendations = recommendations.merge(
        filtered_ratings[['movieId', 'title', 'genres', 'year', 'tmdbId']].drop_duplicates(), 
        on='movieId', how='left'
    )

    recommendations = recommendations[recommendations['genres'].str.contains('|'.join(genres), regex=True)]
    recommendations = recommendations[recommendations['year'].between(startYear, endYear, inclusive='both')]
    recommendations['Rating'] = (recommendations['Rating'] * 2).round() / 2
    
    return recommendations


if __name__ == "__main__":
    genres = json.loads(sys.argv[1])
    years = json.loads(sys.argv[2])
    yearFrom = int(years[0])
    yearTo = int(years[1])
    start = time.time()
    ratings, all_ratings = setup_data()
    filtered_ratings = popular_movies(all_ratings, 100)
    print(f"File setup time: {time.time() - start}", file=sys.stderr)

    user_ids = filtered_ratings['userId'].astype('category').cat.codes
    movie_ids = filtered_ratings['movieId'].astype('category').cat.codes
    rating_values = filtered_ratings['rating']
    
    sparse_matrix = load_npz('../../Recommender/sparse_matrix.npz')

    print(f"Sparse matrix time: {time.time() - start}", file=sys.stderr)

    movie_index_to_id = dict(enumerate(filtered_ratings['movieId'].astype('category').cat.categories))
    movie_id_to_index = {v: k for k, v in movie_index_to_id.items()}

    global_mean = filtered_ratings['rating'].mean()
    movie_means = filtered_ratings.groupby('movieId')['rating'].mean().to_dict()
    topk_inds, topk_sims = kMostSimilar(sparse_matrix, k=50)
    print(f"Kmost time: {time.time() - start}", file=sys.stderr)

    ratings_with_ids, not_found = map_user_ratings_to_movieids(ratings, filtered_ratings)

    recommendations = recommend_movies(ratings_with_ids, all_ratings, sparse_matrix, genres, yearFrom, yearTo, topk_inds, topk_sims, movie_means, global_mean, movie_index_to_id)
    recommendations_json = json.dumps(recommendations.head(20).to_dict(orient='records'))
    print(f"Total time: {time.time() - start}", file=sys.stderr)
    print(recommendations_json)