from flask import Blueprint, flash, render_template, request, jsonify
from flask_login import login_required, current_user


views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    recommendations = []
    if request.method == 'POST':
        genres = request.form.getlist('genres')
        startYear = request.form.get('startYear')
        startYear = int(startYear) if startYear else 1800
        endYear = request.form.get('endYear')
        endYear = int(endYear) if endYear else 2024
        movieCount = request.form.get('movieCount')
        movieCount = int(movieCount) if movieCount else 10
        anySelected = request.form.get('AnySelected') != None
        
        from Recommender.movie_rec import recommend_movies, ratings, movie_similarity_df
        recommendation_list = recommend_movies(ratings, movie_similarity_df, genres, startYear, endYear, anySelected)
        recommendations = recommendation_list.head(movieCount).to_dict(orient='records')
    return render_template('home.html', user=current_user, recommendations=recommendations)