from flask import Flask, render_template, request
from predict import predict_rating
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        # Get data from form
        movie_data = {
            'Title': request.form['title'],
            'Genre': request.form['genre'],
            'Director': request.form['director'],
            'Actor 1': request.form['actor1'],
            'Actor 2': request.form['actor2'],
            'Actor 3': request.form['actor3'],
            'Year': int(request.form['year']),
            'Duration': request.form['duration'] + ' min'
        }
        
        try:
            # Get prediction
            prediction = predict_rating(movie_data)
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
