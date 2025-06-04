# Project Summary: Finding Similar Movies with Spark and LSH

This project implements a system to find similar movies in a large dataset using distributed computing (Apache Spark) and techniques from information retrieval and machine learning, specifically TF-IDF, Inverted Index, Cosine Similarity, and Locality-Sensitive Hashing (LSH).

**Goal:** To identify pairs of movies with high semantic similarity based on their metadata.

**Input Data:**
*   A collection of CSV files (actors, countries, crew, genres, languages, movies, posters, releases, studios, themes) providing metadata about films.
*   The dataset is processed in Google Colab using Python and PySpark.
*   Both a full dataset and a 1% subsampled dataset are used for analysis and testing.

**Key Steps & Techniques:**

1.  **Data Loading and Preparation:**
    *   CSV files are loaded into Spark RDDs.
    *   Data validity is checked during loading.
    *   A subsampled dataset is created for efficiency during development.
    *   Movie metadata is preprocessed and represented as a "Bag of Words" (list of tokens) for each movie ID. Stop words are removed during tokenization.

2.  **TF-IDF Representation:**
    *   **Term Frequency (TF):** Calculated for each token within a movie's token list.
    *   **Inverse Document Frequency (IDF):** Calculated for each unique token across the entire corpus of movies.
    *   **TF-IDF:** The product of TF and IDF, giving a numerical weight to each token in each movie's representation, highlighting important terms.
    *   IDF values are broadcasted for efficient use in TF-IDF calculation.
    *   TF-IDF vectors are converted into sparse vectors (CSR format) for memory efficiency.

3.  **Inverted Index:**
    *   A data structure mapping each unique token to a list of movie IDs that contain that token.
    *   Implemented to quickly find movies sharing common tokens, reducing the number of pairwise comparisons needed.

4.  **Cosine Similarity:**
    *   Measures the angle between two TF-IDF vectors to quantify their similarity.
    *   Calculated as the dot product of the vectors divided by the product of their norms.
    *   Both a standard `cossim` function and a more efficient `fast_cosine_similarity` (using broadcasted TF-IDF weights and norms) are available.

5.  **Locality-Sensitive Hashing (LSH) for Cosine Similarity:**
    *   A technique to approximate nearest neighbor search and improve scalability for similarity comparisons.
    *   **How it works:**
        *   Random hyperplanes are generated in the vector space.
        *   Movie TF-IDF vectors are hashed based on which side of each hyperplane they fall (assigned a binary value, 0 or 1).
        *   The binary hashes are converted to integers, acting as bucket indices in a hash table.
        *   Movies with the same hash (in the same bucket) are likely to be similar.
    *   Random hyperplanes are broadcasted.
    *   Used to significantly reduce the number of movie pairs considered for the final similarity calculation.

6.  **Similarity Detection Pipeline:**
    *   Load and preprocess data (Bag of Words).
    *   Compute TF-IDF values for all movies.
    *   Compute norms of TF-IDF vectors.
    *   Broadcast TF-IDF weights and norms.
    *   Generate random hyperplanes for LSH and broadcast them.
    *   Compute LSH hashes for each movie's TF-IDF vector.
    *   Build a hash table (inverted index where key is hash and value is list of movie IDs).
    *   For each hash bucket:
        *   Retrieve movie IDs in the bucket.
        *   Generate pairs of movies within the bucket.
        *   For each pair, compute the fast cosine similarity using broadcasted weights and norms.
        *   Filter pairs based on a similarity threshold.
        *   Group results by movie pair and collect common tokens.
    *   The final output is a list of movie pairs exceeding a similarity threshold, along with details about their similarity.

**Tools and Libraries:**
*   PySpark for distributed processing.
*   NumPy and SciPy for numerical operations and sparse matrix handling.
*   Pandas for initial data handling (downloading).
*   Kaggle/KaggleHub for dataset access.

**Efficiency Considerations:**
*   Using Apache Spark for distributed processing.
*   Employing TF-IDF to weight terms effectively.
*   Implementing an Inverted Index to quickly find shared features.
*   Utilizing LSH to reduce the number of required pairwise similarity comparisons.
*   Broadcasting large variables (IDF values, hyperplanes, TF-IDF weights, norms) to minimize data transfer between Spark workers.
*   Using sparse vector representations (CSR matrix) to handle high-dimensional data efficiently.

**Evaluation:**
*   The project includes steps for testing and evaluating the system's performance, likely assessing precision and recall of similar movie pairs found.
