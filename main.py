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


def fitness(route, matrix):
    """
    Oblicza całkowitą długość trasy w problemie Komiwojażera

    Parametry:
    route:
        Permutacja indeksów miast np. [0, 1, 2, 3, 4]
    matrix:
        Macierz odległości euklidesowej między miastami

    Zwraca:
        Całkowity koszt trasy
    """
    score = matrix[route[-1]][route[0]]
    for i in range(len(route) - 1):
        score += matrix[route[i]][route[i + 1]]

    return score


def crossing(parent1, parent2):
    """
    Wykonuje krzyżowanie - tworzy nowe dziecko, które:
    - dziedziczy fragment chromosomu z parent1
    - uzupełnia resztę indeksów z parent2 bez powtórzeń

    Parametry:
    parent1:
        Pierwszy rodzic
    parent2:
        Drugi rodzic

    Zwraca:
        Dziecko, czyli chromosom będący skrzyżowaniem obydwu rodziców
    """

    N = len(parent1)
    start, end = sorted(random.sample(range(N), 2))

    child = [None] * N
    child[start:end] = parent1[start:end]

    used = set(child[start:end])

    i = end

    for g in parent2:
        if g not in used:
            if i == N:
                i = 0

            child[i] = g
            i += 1

    return child


def mutation(child, p):
    """
    Mutacja chromosomu - jeśli wylosowana wartość jest mniejsza od p_mutation,
    zamienia miejscami dwa losowe miasta w chromosomie.

    Parametry:
    child:
        Chromosom
    p:
        Prawdopodobieństwo mutacji

    Zwraca:
        Zmutowany lub niezmieniony chromosom
    """

    if random.random() < p:
        N = len(child)
        i, j = random.sample(range(N), 2)

        child[i], child[j] = child[j], child[i]

    return child


def succession(rating, population, child_rating, child_population, population_size):
    """
    Tworzy nowe pokolenie metodą sukcesji. Zachowuje najlepszego osobnika z poprzedniej populacji,
    a pozostałe miejsca w nowej populacji wypełnia najlepszymi potomkami posortowanymi według wartości
    funkcji jakości.

    Parametry:
    rating:
        Lista wartości funkcji jakości dla osobników w starej populacji.
    population:
        Lista chromosomów reprezentujących poprzednie pokolenie.
    child_rating:
        Lista wartości funkcji jakości dla potomków.
    child_population:
        Lista chromosomów potomków po krzyżowaniu i mutacji.
    population_size:
        Liczebność nowej populacji.

    Zwraca:
        Lista chromosomów reprezentujących nowe pokolenie.
        Zawiera najlepszego osobnika ze starej populacji
        oraz population_size - 1 najlepszych potomków.
    """

    best_parent_id = np.argmin(rating)
    best_parent = population[best_parent_id]

    merge = list(zip(child_rating, child_population))
    sorted_merge = sorted(merge)

    best_population = [child for _, child in sorted_merge[:population_size - 1]]
    best_population.append(list(best_parent))

    return best_population


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

# Zapis najlepszego rozwiązania początkowego
best_id = np.argmin(rating)
best_route = list(population[best_id])
best_rating = rating[best_id]

# Historia statystyk do wykresów
history_best = [best_rating]
history_mean = [np.mean(rating)]
history_median = [np.median(rating)]

# Licznik braku poprawy wyniku - do warunku stopu
no_progress = 0

# GŁÓWNA PĘTLA PROGRAMU - ALGORYTM GENETYCZNY - PROBLEM KOMIWOJAŻERA
for gen in range(max_generations):
    # Połączenie populacji z ich oceną
    merge_ratings = zip(rating, population)

    # Sortowanie rosnące po ocenach - im mniejsza tym lepiej
    sorted_ratings = sorted(merge_ratings)

    # Wybranie 50% najlepszych
    parent_size = population_size // 2
    parents = sorted_ratings[:parent_size]

    # Oddzielenie tras od ich ocen i konwersja do listy - pominięcie ratingów
    _, parent_population = zip(*parents)
    parent_population = list(parent_population)

    child_population = []

    while len(child_population) < population_size:

        # Losowy wybór dwóch rodziców
        parent1, parent2 = random.sample(parent_population, 2)

        # Krzyżowanie - brak krzyżowania w wypadku niespełnionego warunku prawdopodobieństwa
        if random.random() < p_crossing:
            child = crossing(parent1, parent2)
        else:
            # Jeśli nie nastąpi krzyżowanie, to kopiujemy pierwszego rodzica
            child = list(parent1)

        # Mutacja
        child = mutation(child, p_mutation)

        # Dodanie dziecka do populacji potomków
        child_population.append(child)

    # Ocena potomków
    child_rating = []

    for child in child_population:
        quality = fitness(child, distance_matrix)
        child_rating.append(quality)

    # Sukcesja
    population = succession(rating, population, child_rating, child_population, population_size)

    # Ocena nowej populacji
    rating = []

    for ch in population:
        quality = fitness(ch, distance_matrix)
        rating.append(quality)

    # Aktualizacja najlepszego wyniku
    # Indeks najlepszego wyniku funkcji jakości (rating)
    current_best_id = np.argmin(rating)
    current_best_rating = rating[current_best_id]

    if current_best_rating < best_rating:
        best_rating = current_best_rating
        best_route = list(population[current_best_id])
        no_progress = 0
    else:
        no_progress += 1

    # Zebranie statystyk do wizualizacji
    history_best.append(best_rating)
    history_mean.append(np.mean(rating))
    history_median.append(np.median(rating))

    # Sprawdzenie warunku stopu
    if no_progress >= stop_after:
        print(f"Zatrzymano po {gen + 1} pokoleniach.")
        break

# WYNIKI KOŃCOWE
print("Najlepszy koszt trasy: ", best_rating)
print("najlepsza trasa - indeksy:")
print(best_route)

# Nazwy miast w kolejności odwiedzania
best_city_names = [cities[i][0] for i in best_route]

print("Najlepsza trasa - miasta:")
print(best_city_names)


