import numpy as np
from scipy.spatial.distance import cdist
import random

# Miasta + ich współrzędne
cities = [
    ("Warszawa", 52.2297, 21.0122),
    ("Kraków", 50.0647, 19.9450),
    ("Wrocław", 51.1079, 17.0385),
    ("Łódź", 51.7592, 19.4560),
    ("Poznań", 52.4064, 16.9252),
    ("Gdańsk", 54.3520, 18.6466),
    ("Szczecin", 53.4289, 14.5530),
    ("Lublin", 51.2465, 22.5684),
    ("Bydgoszcz", 53.1235, 18.0076),
    ("Białystok", 53.1333, 23.1643),
    ("Katowice", 50.2649, 19.0238),
    ("Gdynia", 54.5189, 18.5319),
    ("Częstochowa", 50.7965, 19.1241),
    ("Radom", 51.4025, 21.1471),
    ("Rzeszów", 50.0413, 21.9990),
    ("Toruń", 53.0138, 18.5981),
    ("Sosnowiec", 50.2868, 19.1039),
    ("Kielce", 50.8703, 20.6275),
    ("Gliwice", 50.2976, 18.6766),
    ("Olsztyn", 53.7800, 20.4942),
    ("Bielsko-Biała", 49.8225, 19.0469),
    ("Zabrze", 50.3249, 18.7858),
    ("Bytom", 50.3480, 18.9328),
    ("Zielona Góra", 51.9355, 15.5064),
    ("Rybnik", 50.0971, 18.5418),
    ("Ruda Śląska", 50.2584, 18.8563),
    ("Opole", 50.6721, 17.9253),
    ("Tychy", 50.1372, 18.9664),
    ("Gorzów Wielkopolski", 52.7368, 15.2288),
    ("Dąbrowa Górnicza", 50.3182, 19.2374),
    ("Elbląg", 54.1522, 19.4088),
    ("Płock", 52.5468, 19.7064),
    ("Koszalin", 54.1944, 16.1722),
    ("Tarnów", 50.0138, 20.9870),
    ("Włocławek", 52.6482, 19.0678),
    ("Chorzów", 50.3058, 18.9742),
    ("Wałbrzych", 50.7714, 16.2843),
    ("Piaseczno", 52.0814, 21.0240),
    ("Kalisz", 51.7611, 18.0910),
    ("Legnica", 51.2101, 16.1619),
    ("Grudziądz", 53.4841, 18.7537),
    ("Jaworzno", 50.2053, 19.2750),
    ("Słupsk", 54.4641, 17.0287),
    ("Jastrzębie-Zdrój", 49.9554, 18.5748),
    ("Nowy Sącz", 49.6218, 20.6971),
    ("Jelenia Góra", 50.8997, 15.7290),
    ("Siedlce", 52.1677, 22.2900),
    ("Mysłowice", 50.2075, 19.1667),
    ("Piła", 53.1515, 16.7378),
    ("Ostrów Wielkopolski", 51.6550, 17.8069)
]

# Inicjalizacja parametrów populacji
N = len(cities)  # ilość miast
population_size = 100  # liczba chromosomów (tras) w każdej iteracji
p_crossing = 0.9  # prawdopodobieństwo krzyżowania
p_mutation = 0.05  # prawodpodobieństwo mutacji
max_generations = 300  # maksymalna liczba generacji
stop_after = 50  # warunek stopu - liczba generacji bez poprawy wyniku, po których algorytm kończy działanie


def fitness(route, distance_matrix):
    """
    Oblicza całkowitą długość trasy w problemie Komiwojażera

    Parametry:
    route:
        Permutacja indeksów miast np. [0, 1, 2, 3, 4]
    distance_matrix:
        Macierz odległości euklidesowej między miastami

    Zwraca:
        Całkowity koszt trasy
    """
    score = distance_matrix[route[-1]][route[0]]
    for i in range(len(route) - 1):
        score += distance_matrix[route[i]][route[i + 1]]


# Wyciągnięcie samych współrzędnych
coords = np.array([(lat, lon) for _, lat, lon in cities])

# Macierz odległości euklidesowej
distance_matrix = cdist(coords, coords, metric="euclidean")

population = []
rating = []

# Tworzenie tras i obliczanie ich funkcji jakości
for i in range(population_size):
    ch = random.sample(range(N), N)
    population.append(ch)

    quality = fitness(ch, distance_matrix)
    rating.append(quality)

