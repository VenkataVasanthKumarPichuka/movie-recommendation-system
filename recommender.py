import pickle

movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x:x[1])[1:6]

    recommended_movies = []

    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


# Test locally
if __name__ == "__main__":
    movie_name = input("Enter Movie Name: ")
    recommendations = recommend(movie_name)

    print("\nTop 5 Recommendations:")
    for movie in recommendations:
        print(movie)
