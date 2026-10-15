import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("data/products.csv")


# ============================================================
# CLEAN DATA
# ============================================================

df["Product Name"] = df["Product Name"].fillna("").astype(str)
df["Description"] = df["Description"].fillna("").astype(str)
df["Category"] = df["Category"].fillna("").astype(str)


# ============================================================
# CREATE COMBINED FEATURES
# Product Name + Description + Category
# ============================================================

df["combined_features"] = (
    df["Product Name"] + " "
    + df["Description"] + " "
    + df["Category"]
)


# ============================================================
# TF-IDF
# ============================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    df["combined_features"]
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend(product_name, num_recommendations=5):

    product_name = product_name.strip()

    # --------------------------------------------------------
    # Check product
    # --------------------------------------------------------

    matching_products = df[
        df["Product Name"].str.lower()
        == product_name.lower()
    ]

    if matching_products.empty:
        return []


    # --------------------------------------------------------
    # Get selected product index
    # --------------------------------------------------------

    idx = matching_products.index[0]


    # --------------------------------------------------------
    # Get similarity scores
    # --------------------------------------------------------

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )


    # --------------------------------------------------------
    # Sort highest similarity first
    # --------------------------------------------------------

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )


    recommendations = []


    # --------------------------------------------------------
    # Create recommendations
    # --------------------------------------------------------

    for product_index, score in similarity_scores:

        # Skip the selected product itself
        if product_index == idx:
            continue


        product = df.iloc[product_index]


        similarity_percentage = round(
            float(score) * 100,
            1
        )


        recommendations.append({

            "name": product["Product Name"],

            "description": product["Description"],

            "category": product["Category"],

            "price": float(product["Price"]),

            "rating": float(product["Rating"]),

            "image": product["Image"],

            "similarity": similarity_percentage

        })


        # Stop after required number
        if len(recommendations) >= num_recommendations:
            break


    return recommendations


# ============================================================
# TEST SYSTEM
# ============================================================

if __name__ == "__main__":

    selected_product = "Wireless Mouse"

    results = recommend(
        selected_product,
        5
    )


    print()
    print("==========================================")
    print(" AI PRODUCT RECOMMENDATION SYSTEM")
    print("==========================================")
    print()

    print(
        "Selected Product:",
        selected_product
    )

    print()

    print("Recommended Products:")

    print("------------------------------------------")


    if not results:

        print(
            "No recommendations found."
        )


    else:

        for item in results:

            print(
                f"{item['name']} "
                f"| Similarity: "
                f"{item['similarity']}%"
            )


    print()