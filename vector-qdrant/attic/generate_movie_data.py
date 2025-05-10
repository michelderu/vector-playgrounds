import requests
import os
import time
import json

API_KEY = os.getenv("TMDB_API_KEY")

movies = []
pages = 50 # Number of pages to process, every page retrieves 20 movies
for page in range(0, pages):  
    print (f"Processing page {page+1} of {pages}")
    url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page={page+1}"
    response = requests.get(url)
    data = response.json()

    for movie in data['results']:
        movie_data = []
        movie_data.append({
            'title': movie['title'],
            'overview': movie['overview'],
            'poster_url': 'https://image.tmdb.org/t/p/w500' + movie['poster_path'] if movie['poster_path'] else None,
            'vote_average': movie['vote_average']
        })
        movies.append(movie_data)

    time.sleep(0.1)  # Sleep for 0.1 seconds to limit to 10 requests per second

# Save the movies to a JSON file
with open('movies.json', 'w') as f:
    json.dump(movies, f)

