from flask import Blueprint, flash, render_template, request, jsonify
from flask_login import login_required, current_user


views = Blueprint('views', __name__)

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    recommendations = []
    if request.method == 'POST':
        from Recommender.movie_rec import recommend_movies, ratings, movie_similarity_df
        
        recommendation_list = recommend_movies(ratings, movie_similarity_df)
        recommendations = recommendation_list.head(10).to_dict(orient='records')
    return render_template('home.html', user=current_user, recommendations=recommendations)