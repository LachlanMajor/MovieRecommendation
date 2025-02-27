from flask import Blueprint, flash, render_template, request, jsonify
from flask_login import login_required, current_user
from .models import Recommendation, db

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

        all_recommendations = recommendation_list.to_dict(orient='records')
        saved_movies = {rec.title for rec in Recommendation.query.filter_by(user_id=current_user.id).all()}
        filtered = [rec for rec in all_recommendations if rec['Movie'] not in saved_movies]


        recommendations = filtered[:movieCount]

    return render_template('home.html', user=current_user, recommendations=recommendations)

@views.route('/save', methods=['POST'])
@login_required
def save():
    print('reached save')
    data = request.get_json()
    movie = data.get('Movie')
    rating = data.get('Expected Rating')
    genres = data.get('genres')
    year = data.get('year')

    new_rec = Recommendation(
        title = movie,
        expectedRating = rating,
        genres = genres,
        year = int(float(year)),
        user_id = current_user.id
    )
    db.session.add(new_rec)
    db.session.commit()
    return jsonify({"success": True})

@views.route('/delete', methods=['DELETE'])
@login_required
def delete():
    print('reached delete')
    data = request.get_json()
    movie = data.get('Movie')
    rec = Recommendation.query.filter_by(title=movie).first()
    db.session.delete(rec)
    db.session.commit()
    return render_template("saved.html")
    

@views.route('/save_recommendation')
@login_required
def saved_recommendations():
    user_id = current_user.id
    saved_movies = Recommendation.query.filter_by(user_id=user_id).all()
    
    return render_template("saved.html", saved_movies=saved_movies)