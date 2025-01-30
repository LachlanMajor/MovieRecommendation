from flask import Blueprint, flash, render_template, request, jsonify
from flask_login import login_required, current_user


views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    recommendations = []
    if request.method == 'POST':
        genres = request.form.getlist('genres')
        start_year = request.form.get('startYear')
        end_year = request.form.get('endYear')
        start_year = int(start_year) if start_year else 1981
        end_year = int(end_year) if end_year else 1981
        
        from Recommender.movie_rec import recommend_movies, ratings, movie_similarity_df
        
        recommendation_list = recommend_movies(ratings, movie_similarity_df, genres, start_year, end_year)
        recommendations = recommendation_list.head(10).to_dict(orient='records')
    return render_template('home.html', user=current_user, recommendations=recommendations)