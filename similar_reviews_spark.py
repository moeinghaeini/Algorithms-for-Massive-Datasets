import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split, udf
from pyspark.sql.types import ArrayType, StringType, SetType
from itertools import combinations

# Define a set of common English stop words
# (This is a basic list, a more comprehensive one might be needed for better results)
STOP_WORDS = set([
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
])

def preprocess_text(text):
    if text is None:
        return []
    # Lowercase
    text = text.lower()
    # Remove punctuation (and numbers for simplicity, though numbers might be relevant in some contexts)
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    # Tokenize by splitting whitespace
    words = text.split()
    # Remove stop words and create a set of unique words
    processed_words = set(word for word in words if word not in STOP_WORDS and len(word) > 1) # also filter out single char tokens
    return list(processed_words) # Return list for Spark UDF compatibility with ArrayType(StringType())

# UDF for text preprocessing
preprocess_text_udf = udf(preprocess_text, ArrayType(StringType()))

def calculate_jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 0.0
    if not set1 or not set2: # if one is empty, and the other is not
        return 0.0

    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    if union_size == 0: # handles case where both sets are empty after processing, though caught above
        return 0.0
    return float(intersection_size) / union_size

def main():
    spark = SparkSession.builder.appName("SimilarBookReviews").getOrCreate()
    sc = spark.sparkContext # Get SparkContext

    # --- Configuration ---
    # Path to your Books_rating.csv file
    # dataset_path = "path/to/your/Books_rating.csv" # Replace with the actual path in your Spark environment
    # The path is known from the download step, but for a real Spark cluster, this might be an HDFS path
    # For now, let's assume the script will be run in an environment where this path is accessible.
    # We'll need to make sure the user replaces this placeholder or the path is correctly passed.
    # For the purpose of this tool, I will use the path we know.
    dataset_path = "/home/jules/.cache/kagglehub/datasets/mohamedbakhet/amazon-books-reviews/versions/1/Books_rating.csv"
    similarity_threshold = 0.7 # Jaccard similarity threshold

    print(f"Starting similar book review detection from: {dataset_path}")
    print(f"Similarity threshold: {similarity_threshold}")

    # a. Load Data
    # Only select 'Id' and 'review/text'. Handle potential missing values.
    try:
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(dataset_path)
        # Ensure correct column names if they contain special characters like '/'
        # Spark might replace them, e.g., 'review/text' might become 'review_text'
        # Let's check the schema
        # df.printSchema() # You would uncomment this to debug column names in a real Spark session

        # Explicitly select and rename if necessary. Assuming 'Id' and 'review/text' are the correct names.
        # If 'review/text' is problematic, you might need to escape it or find how Spark names it.
        # For example, df.select(col("`review/text`").alias("review_text"), "Id")
        # For now, assume direct naming works or Spark handles it.
        reviews_df = df.select(col("Id").alias("review_id"), col("`review/text`").alias("review_text")).na.drop(subset=["review_text"])
        reviews_df = reviews_df.limit(1000) # FOR DEVELOPMENT: Limit data size to speed up testing. Remove for full run.
        print(f"Successfully loaded and selected Id and review/text. Count: {reviews_df.count()}")

    except Exception as e:
        print(f"Error loading data: {e}")
        spark.stop()
        return

    # b. Preprocessing & Tokenization
    # Apply UDF to create a new column 'processed_words' (as a list, convert to set later for Jaccard)
    tokenized_reviews_df = reviews_df.withColumn("processed_words", preprocess_text_udf(col("review_text")))

    # Keep only review_id and the set of processed_words for further processing
    # Convert list from UDF to set for efficient Jaccard calculation later
    # We'll do this conversion when we collect the data or right before Jaccard
    # For now, keep as ArrayType for Spark DataFrame operations
    processed_rdd = tokenized_reviews_df.select("review_id", "processed_words").rdd \
        .map(lambda row: (row.review_id, set(row.processed_words))) \
        .filter(lambda x: len(x[1]) > 0) # Filter out reviews that become empty after preprocessing

    # Cache this RDD as it will be used multiple times
    processed_rdd.cache()
    print(f"Preprocessing and tokenization complete. RDD count: {processed_rdd.count()}")
    # print("Sample of processed RDD:", processed_rdd.take(5))


    # c. Generate Candidate Pairs (Inverted Index approach)
    # Map 1: (word, review_id)
    word_to_review_id_rdd = processed_rdd.flatMap(lambda x: [(word, x[0]) for word in x[1]])
    # print("Sample of word_to_review_id_rdd:", word_to_review_id_rdd.take(10))

    # Reduce 1 (Group by word): (word, [review_id1, review_id2, ...])
    word_to_review_ids_list_rdd = word_to_review_id_rdd.groupByKey().mapValues(list)
    # print("Sample of word_to_review_ids_list_rdd:", word_to_review_ids_list_rdd.take(5))

    # Map 2 (Generate pairs): ([review_id_pair1, review_id_pair2, ...])
    # For each list of review_ids sharing a common word, generate unique pairs
    # Ensure review_id1 < review_id2 to avoid duplicate pairs like (B,A) if (A,B) exists
    # and to avoid pairing a review with itself.
    candidate_pairs_rdd = word_to_review_ids_list_rdd.flatMap(
        lambda x: [tuple(sorted(pair)) for pair in combinations(x[1], 2)]
    ).distinct() # Get unique pairs

    # Cache candidate pairs as they are crucial for the next step
    candidate_pairs_rdd.cache()
    print(f"Candidate pair generation complete. Number of candidate pairs: {candidate_pairs_rdd.count()}")
    # print("Sample of candidate_pairs_rdd:", candidate_pairs_rdd.take(10))

    # d. Calculate Jaccard Similarity for Candidate Pairs
    # We need the word sets for each review to calculate Jaccard.
    # Broadcast the processed_rdd data (or a map version of it) if it's not too large,
    # or join candidate_pairs_rdd with processed_rdd.
    # Joining is more robust for larger datasets.

    # Create RDDs for joining: (review_id, word_set)
    review_word_sets_rdd = processed_rdd # This is already (review_id, word_set)

    # Join candidate pairs with word sets
    # pair: (id1, id2)
    # Need to join twice to get word sets for both ids in the pair

    # (id1, (id2, word_set1))
    pairs_with_set1 = candidate_pairs_rdd.map(lambda pair: (pair[0], pair[1])) \
        .join(review_word_sets_rdd) \
        .map(lambda x: (x[1][0], (x[0], x[1][1]))) # (id2, (id1, word_set1))

    # (id2, ((id1, word_set1), word_set2))
    # Then map to ((id1, id2), (word_set1, word_set2))
    joined_pairs_with_sets = pairs_with_set1.join(review_word_sets_rdd) \
        .map(lambda x: ((x[1][0][0], x[0]), (x[1][0][1], x[1][1]))) # ((id1, id2), (set1, set2))

    # Calculate Jaccard similarity
    similarities_rdd = joined_pairs_with_sets.map(
        lambda x: (x[0], calculate_jaccard_similarity(x[1][0], x[1][1]))
    )
    # print("Sample of similarities_rdd:", similarities_rdd.take(10))

    # e. Filter and Output
    highly_similar_pairs_rdd = similarities_rdd.filter(lambda x: x[1] >= similarity_threshold)

    print(f"Found {highly_similar_pairs_rdd.count()} pairs with similarity >= {similarity_threshold}")

    # Collect and print results (for smaller result sets)
    # For very large result sets, you would save this to HDFS or another storage system
    results = highly_similar_pairs_rdd.collect()
    if results:
        print(f"\n--- Highly Similar Review Pairs (Similarity >= {similarity_threshold}) ---")
        for pair, similarity in results:
            print(f"Review Pair: {pair[0]} - {pair[1]}, Similarity: {similarity:.4f}")
    else:
        print("No pairs found above the similarity threshold.")

    # To retrieve actual review text for the similar pairs (optional, for inspection):
    # You could join `highly_similar_pairs_rdd` back with `reviews_df` (or `processed_rdd`)
    # This part is left as an extension if needed.

    print("--- Spark Job Finished ---")
    spark.stop()

if __name__ == "__main__":
    main()

# Instructions for running (example):
# 1. Ensure you have Spark installed and configured.
# 2. Place `Books_rating.csv` in a path accessible by Spark (e.g., HDFS or a local path on all nodes if running locally).
#    Update `dataset_path` in the script if it's different from the default cache path.
# 3. Submit the script using spark-submit:
#    spark-submit \
#      --master local[*] \  # Or your cluster's master URL
#      similar_reviews_spark.py
#
# Notes on the script:
# - A basic stop word list is included. For better results, use a more comprehensive list or a library like NLTK for stop word removal.
# - Text preprocessing is basic. More advanced techniques (stemming, lemmatization) could improve similarity detection.
# - The candidate pair generation uses an inverted index approach. For extremely large datasets,
#   more advanced techniques like Locality Sensitive Hashing (LSH) would be more efficient at reducing the
#   number of pairs for direct Jaccard calculation. This script provides a foundational MapReduce-style approach.
# - Error handling for file not found or incorrect CSV format is basic.
# - The `limit(1000)` on `reviews_df` is for development/testing. Remove it to process the entire dataset.
#   Processing the full dataset will take significant time and resources.
# - The Jaccard calculation `calculate_jaccard_similarity` handles empty sets to avoid division by zero.
# - The script assumes 'Id' is a unique identifier for reviews.
# - Spark column name handling for special characters (like '/') is addressed by backticks: col("`review/text`").
# - Caching (.cache()) is used for RDDs that are reused, which is important for performance in Spark.
# - The final output prints to console. For large outputs, writing to a file (e.g., on HDFS) is recommended.
#   Example: highly_similar_pairs_rdd.saveAsTextFile("path/to/output_directory")
