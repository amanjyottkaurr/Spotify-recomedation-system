import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import joblib


# =========================================================
# 1. LOAD DATASET
# =========================================================

file_name = "spotify_recommendation_dataset.csv"

df = pd.read_csv(file_name)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =========================================================
# 2. REMOVE DUPLICATES
# =========================================================

df = df.drop_duplicates().copy()

print("After removing duplicates:", df.shape)


# =========================================================
# 3. CLEAN CATEGORICAL COLUMNS
# =========================================================

categorical_columns = [
    "track_name",
    "artist_name",
    "album_name",
    "genre",
    "language",
    "mood"
]

for column in categorical_columns:
    df[column] = df[column].astype("string").str.strip()


# =========================================================
# 4. CLEAN AUDIO FEATURES
# =========================================================

audio_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

for column in audio_features:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = df[column].fillna(
        df[column].median()
    )


# =========================================================
# 5. CREATE UNIQUE SONG DATASET
# =========================================================

songs = (
    df.drop_duplicates(subset="track_id")
      .reset_index(drop=True)
)

print("Unique songs:", len(songs))


# =========================================================
# 6. CONTENT-BASED MODEL
# =========================================================

scaler = StandardScaler()

content_features = scaler.fit_transform(
    songs[audio_features]
)

content_similarity = cosine_similarity(
    content_features
)

content_index = pd.Series(
    songs.index,
    index=songs["track_id"]
)


# =========================================================
# 7. COLLABORATIVE FILTERING DATA
# =========================================================

ratings = df[
    ["user_id", "track_id", "user_rating"]
].copy()

ratings["user_rating"] = pd.to_numeric(
    ratings["user_rating"],
    errors="coerce"
)

ratings = ratings.dropna(
    subset=["user_id", "track_id", "user_rating"]
)


# =========================================================
# 8. CREATE USER-ITEM MATRIX
# =========================================================

user_item = ratings.pivot_table(
    index="user_id",
    columns="track_id",
    values="user_rating",
    aggfunc="mean"
)

user_item = user_item.fillna(0)

print("User-item matrix:", user_item.shape)


# =========================================================
# 9. COLLABORATIVE SIMILARITY
# =========================================================

collaborative_similarity = cosine_similarity(
    user_item.T
)

collaborative_index = pd.Series(
    range(len(user_item.columns)),
    index=user_item.columns
)


# =========================================================
# 10. FIND COMMON SONGS
# =========================================================

common_tracks = list(
    set(songs["track_id"])
    & set(user_item.columns)
)

print("Common songs:", len(common_tracks))


# =========================================================
# 11. ALIGN BOTH MODELS
# =========================================================

common_content_indices = [
    content_index[track]
    for track in common_tracks
]

common_collaborative_indices = [
    collaborative_index[track]
    for track in common_tracks
]

aligned_content_similarity = content_similarity[
    np.ix_(
        common_content_indices,
        common_content_indices
    )
]

aligned_collaborative_similarity = collaborative_similarity[
    np.ix_(
        common_collaborative_indices,
        common_collaborative_indices
    )
]

print(
    "Aligned content matrix:",
    aligned_content_similarity.shape
)

print(
    "Aligned collaborative matrix:",
    aligned_collaborative_similarity.shape
)


# =========================================================
# 12. HYBRID MODEL
# =========================================================

CONTENT_WEIGHT = 0.5
COLLABORATIVE_WEIGHT = 0.5


def hybrid_recommend(song_id, number=10):

    if song_id not in common_tracks:
        return []

    song_position = common_tracks.index(song_id)

    content_scores = (
        aligned_content_similarity[song_position]
    )

    collaborative_scores = (
        aligned_collaborative_similarity[song_position]
    )

    hybrid_scores = (
        CONTENT_WEIGHT * content_scores
        +
        COLLABORATIVE_WEIGHT * collaborative_scores
    )

    results = []

    for i, track_id in enumerate(common_tracks):

        if track_id == song_id:
            continue

        results.append(
            (
                track_id,
                hybrid_scores[i]
            )
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results[:number]


# =========================================================
# 13. DISPLAY RECOMMENDATIONS
# =========================================================

def show_recommendations(song_id, number=10):

    recommendations = hybrid_recommend(
        song_id,
        number
    )

    if not recommendations:
        print("Song not found in the hybrid model.")
        return

    selected_song = songs[
        songs["track_id"] == song_id
    ]

    if len(selected_song) > 0:

        selected_song = selected_song.iloc[0]

        print("\n----------------------------------")
        print("Selected Song")
        print("----------------------------------")

        print(
            selected_song["track_name"],
            "-",
            selected_song["artist_name"]
        )

    print("\nRecommended Songs")
    print("----------------------------------")

    for i, (track_id, score) in enumerate(
        recommendations,
        start=1
    ):

        song = songs[
            songs["track_id"] == track_id
        ].iloc[0]

        print(
            f"{i}. "
            f"{song['track_name']} - "
            f"{song['artist_name']} "
            f"| Genre: {song['genre']} "
            f"| Score: {score:.3f}"
        )


# =========================================================
# 14. TEST RECOMMENDATION
# =========================================================

test_song = common_tracks[0]

show_recommendations(
    test_song,
    10
)


# =========================================================
# 15. PRECISION@10
# =========================================================

def precision_at_k(recommended, relevant, k=10):

    recommended = recommended[:k]

    if len(recommended) == 0:
        return 0

    hits = len(
        set(recommended) & set(relevant)
    )

    return hits / len(recommended)


# =========================================================
# 16. RECALL@10
# =========================================================

def recall_at_k(recommended, relevant, k=10):

    recommended = recommended[:k]

    if len(relevant) == 0:
        return 0

    hits = len(
        set(recommended) & set(relevant)
    )

    return hits / len(relevant)


# =========================================================
# 17. NDCG@10
# =========================================================

def ndcg_at_k(recommended, relevant, k=10):

    recommended = recommended[:k]

    dcg = 0

    for i, item in enumerate(recommended):

        if item in relevant:

            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(
        len(relevant),
        k
    )

    if ideal_hits == 0:
        return 0

    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(ideal_hits)
    )

    return dcg / idcg


# =========================================================
# 18. CREATE POSITIVE INTERACTIONS
# =========================================================

positive_data = df[
    df["user_rating"] >= 4
][
    ["user_id", "track_id", "user_rating"]
].dropna()

print(
    "\nPositive interactions:",
    len(positive_data)
)


# =========================================================
# 19. ACTIVE USERS
# =========================================================

user_counts = (
    positive_data["user_id"]
    .value_counts()
)

active_users = user_counts[
    user_counts >= 3
].index

evaluation_data = positive_data[
    positive_data["user_id"].isin(
        active_users
    )
].copy()

print(
    "Active users:",
    len(active_users)
)


# =========================================================
# 20. CREATE TEST SET
# =========================================================

test_data = (
    evaluation_data
    .groupby("user_id")
    .sample(
        n=1,
        random_state=42
    )
)

print(
    "Test interactions:",
    len(test_data)
)


# =========================================================
# 21. EVALUATE HYBRID MODEL
# =========================================================

precision_scores = []
recall_scores = []
ndcg_scores = []


for _, row in test_data.iterrows():

    user_id = row["user_id"]
    actual_song = row["track_id"]

    user_history = evaluation_data[
        evaluation_data["user_id"] == user_id
    ]

    previous_songs = user_history[
        user_history["track_id"] != actual_song
    ]["track_id"].tolist()

    previous_songs = [
        song
        for song in previous_songs
        if song in common_tracks
    ]

    if len(previous_songs) == 0:
        continue

    seed_song = previous_songs[0]

    recommendations = hybrid_recommend(
        seed_song,
        10
    )

    recommended_ids = [
        item[0]
        for item in recommendations
    ]

    relevant = [actual_song]

    precision_scores.append(
        precision_at_k(
            recommended_ids,
            relevant,
            10
        )
    )

    recall_scores.append(
        recall_at_k(
            recommended_ids,
            relevant,
            10
        )
    )

    ndcg_scores.append(
        ndcg_at_k(
            recommended_ids,
            relevant,
            10
        )
    )


# =========================================================
# 22. FINAL METRICS
# =========================================================

precision = (
    np.mean(precision_scores)
    if precision_scores
    else 0
)

recall = (
    np.mean(recall_scores)
    if recall_scores
    else 0
)

ndcg = (
    np.mean(ndcg_scores)
    if ndcg_scores
    else 0
)


print("\n===================================")
print("HYBRID MODEL PERFORMANCE")
print("===================================")

print(
    f"Precision@10: {precision:.4f}"
)

print(
    f"Recall@10:    {recall:.4f}"
)

print(
    f"NDCG@10:      {ndcg:.4f}"
)


# =========================================================
# 23. SAVE RESULTS
# =========================================================

results = pd.DataFrame({
    "Model": ["Hybrid"],
    "Precision@10": [precision],
    "Recall@10": [recall],
    "NDCG@10": [ndcg]
})

results.to_csv(
    "hybrid_results.csv",
    index=False
)

print("\nResults saved as hybrid_results.csv")


# =========================================================
# 24. SAVE MODEL FILES
# =========================================================

joblib.dump(
    scaler,
    "hybrid_scaler.pkl"
)

joblib.dump(
    aligned_content_similarity,
    "hybrid_content_similarity.pkl"
)

joblib.dump(
    aligned_collaborative_similarity,
    "hybrid_collaborative_similarity.pkl"
)

songs.to_csv(
    "hybrid_songs.csv",
    index=False
)

joblib.dump(
    common_tracks,
    "hybrid_common_tracks.pkl"
)

print("Model files saved successfully!")

print("\n===================================")
print("MODEL 3 COMPLETED SUCCESSFULLY!")
print("===================================")