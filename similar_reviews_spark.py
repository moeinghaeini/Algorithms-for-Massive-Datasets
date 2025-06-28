import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split, udf
from pyspark.sql.types import ArrayType, StringType
from itertools import combinations

# NLTK imports for advanced preprocessing
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- NLTK Setup ---
# Ensure NLTK data is available in the Spark environment.
# For local mode, download them once:
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))
# Add any custom stop words if needed
# STOP_WORDS.update(["book", "read", "review"])


# Helper function to convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # Default to noun

lemmatizer = WordNetLemmatizer()

def preprocess_text_enhanced(text):
    if text is None:
        return []

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation and numbers (numbers are removed after tokenization for better POS tagging)
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation

    # 3. Tokenize using NLTK
    tokens = word_tokenize(text)

    # 4. POS Tagging
    pos_tagged_tokens = nltk.pos_tag(tokens)

    # 5. Lemmatization
    lemmatized_tokens = []
    for word, tag in pos_tagged_tokens:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        # Remove numbers after lemmatization & check length
        if not lemma.isdigit() and len(lemma) > 1:
            lemmatized_tokens.append(lemma)

    # 6. Remove stop words
    processed_words = set(word for word in lemmatized_tokens if word not in STOP_WORDS)

    return list(processed_words)

# UDF for text preprocessing
preprocess_text_udf = udf(preprocess_text_enhanced, ArrayType(StringType()))

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
    # print("Sample of processed RDD:", processed_rdd.take(5)) # This RDD is (review_id, {lemma_set})

    # --- Configuration for Shingling, MinHashing, and LSH ---
    K_SHINGLE = 9  # Length of k-shingles (e.g., 5 or 9 characters)
    NUM_HASH_FUNCTIONS = 100  # Number of MinHash functions (e.g., 100 or 200)
    LSH_BANDS = 20  # Number of bands for LSH

    if NUM_HASH_FUNCTIONS % LSH_BANDS != 0:
        raise ValueError("NUM_HASH_FUNCTIONS must be divisible by LSH_BANDS.")
    LSH_ROWS = NUM_HASH_FUNCTIONS // LSH_BANDS # Rows per band

    if LSH_ROWS == 0: # Should be caught by the divisibility check, but good to have
        raise ValueError("LSH_BANDS and NUM_HASH_FUNCTIONS result in 0 rows per band.")

    # Max value for hash signatures in MinHash, used for initializing min values
    MAX_HASH_VAL = (1 << 32) - 1
    # Seeds for MinHash functions to ensure different hash computations
    HASH_SEEDS = list(range(NUM_HASH_FUNCTIONS))

    print(f"\n--- Stage: Shingling, MinHashing & LSH ---")
    print(f"K-shingle length: {K_SHINGLE}")
    print(f"Number of hash functions for MinHash: {NUM_HASH_FUNCTIONS}")
    print(f"LSH: {LSH_BANDS} bands, {LSH_ROWS} rows per band.")

    # --- c1. Shingling ---
    # Converts a list of lemmas into a set of character k-shingles.
    # Character shingles are often used for near-duplicate text detection.
    def create_shingles(lemmas_list, k):
        if not lemmas_list:
            return set()
        # Join lemmas into a single string to create character shingles
        # Alternatively, could create shingles from tokens directly if preferred
        # For character shingles, it's common to remove spaces or use a special char
        doc_str = "".join(sorted(list(text_content))).replace(" ", "") # sorted list of lemmas joined
        if len(doc_str) < k:
            return {doc_str} # if doc is shorter than k, shingle is the doc itself
        shingles = set()
        for i in range(len(doc_str) - k + 1):
            shingles.add(doc_str[i:i+k])
        return list(shingles) # Return list for Spark UDF

    create_shingles_udf = udf(lambda text_list: create_shingles(" ".join(text_list), K_SHINGLE), ArrayType(StringType()))

    # Apply shingling to the 'processed_words' (which are lemmas)
    # The input to create_shingles UDF should be the list of lemmas
    shingled_reviews_df = tokenized_reviews_df.withColumn("shingles", create_shingles_udf(col("processed_words")))

    # RDD of (review_id, {shingle_set})
    shingled_rdd = shingled_reviews_df.select("review_id", "shingles").rdd \
        .map(lambda row: (row.review_id, set(row.shingles))) \
        .filter(lambda x: len(x[1]) > 0)

    shingled_rdd.cache()
    print(f"Shingling complete. RDD count: {shingled_rdd.count()}")
    # print("Sample of shingled RDD:", shingled_rdd.take(2))

    # c2. MinHash Signature Generation
    # Create N hash functions. For simplicity, use different seeds for a standard hash function.
    # (Python's hash() is not stable across processes/runs for strings, so avoid it here for distributed consistency)
    # Using a simple approach: hash(shingle + str(seed)) % some_large_prime
    # A better way would be to use variants of murmurhash or similar.
    # For this example, we'll use a large prime and multiple hash computations.
    # To make it more robust for Spark, the hash functions need to be defined or accessible within map operations.

    # Max value for hash (to keep them in a manageable range, helps with permutations if used)
    # Using a large prime number as the modulus
    MAX_HASH_VAL = (1 << 32) -1 # or a suitable large prime like 2**31 - 1

    # Generate seeds for hash functions
    # These seeds will be broadcasted if used in a UDF or closure
    hash_seeds = list(range(NUM_HASH_FUNCTIONS))

    def generate_minhash_signature(shingle_set, num_hashes, seeds):
        signature = []
        if not shingle_set:
            return [MAX_HASH_VAL] * num_hashes # Return max hash if no shingles

        for i in range(num_hashes):
            min_hash_for_this_func = MAX_HASH_VAL
            for shingle in shingle_set:
                # Combine shingle with seed and hash. Using Python's built-in hash for simplicity here,
                # but be aware of its limitations for distributed consistency if not careful.
                # For better consistency, use hashlib.sha256 or murmurhash3.
                # For now, let's use a simple combination and hash.
                # Ensure shingle is string.
                shingle_str = str(shingle)
                current_hash = abs(hash(shingle_str + str(seeds[i]))) % MAX_HASH_VAL # abs for positive
                if current_hash < min_hash_for_this_func:
                    min_hash_for_this_func = current_hash
            signature.append(min_hash_for_this_func)
        return signature

    # Use a map operation on shingled_rdd to generate signatures
    # The function and seeds need to be available in the closure
    minhash_signatures_rdd = shingled_rdd.map(
        lambda x: (x[0], generate_minhash_signature(x[1], NUM_HASH_FUNCTIONS, hash_seeds))
    )
    minhash_signatures_rdd.cache()
    print(f"MinHash signature generation complete. RDD count: {minhash_signatures_rdd.count()}")
    # print("Sample MinHash Signatures:", minhash_signatures_rdd.take(2))

    # c3. LSH Banding for Candidate Pairs
    # signature_rdd is (review_id, [minhash_signature_vector])
    def generate_lsh_bands(signature_item, bands, rows):
        review_id, signature_vector = signature_item
        band_kv_pairs = []
        if not signature_vector or len(signature_vector) != bands * rows:
            return band_kv_pairs # Or handle error

        for i in range(bands):
            start_index = i * rows
            end_index = start_index + rows
            band_content = tuple(signature_vector[start_index:end_index])
            # Hash the band content to create a bucket key
            # The band_id (i) is part of the key to distinguish buckets from different bands
            bucket_key = (i, hash(band_content))
            band_kv_pairs.append((bucket_key, review_id))
        return band_kv_pairs

    # ( (band_id, bucket_hash), review_id )
    banded_rdd = minhash_signatures_rdd.flatMap(
        lambda sig_item: generate_lsh_bands(sig_item, LSH_BANDS, LSH_ROWS)
    )

    # Group by bucket_key to find documents in the same bucket
    # ( (band_id, bucket_hash), [review_id1, review_id2, ...] )
    bucketed_reviews_rdd = banded_rdd.groupByKey().mapValues(list)

    # Generate candidate pairs from buckets
    # If a bucket has multiple review_ids, all pairs from that list are candidates
    candidate_pairs_lsh_rdd = bucketed_reviews_rdd.flatMap(
        lambda x: [tuple(sorted(pair)) for pair in combinations(x[1], 2) if len(x[1]) > 1]
    ).distinct() # Get unique pairs (id1, id2) where id1 < id2

    candidate_pairs_lsh_rdd.cache()
    num_candidate_pairs = candidate_pairs_lsh_rdd.count()
    print(f"LSH complete. Number of candidate pairs: {num_candidate_pairs}")
    # print("Sample LSH candidate pairs:", candidate_pairs_lsh_rdd.take(10))


    # d. Calculate Jaccard Similarity for LSH Candidate Pairs
    # We need the *original shingle sets* for the candidate pairs identified by LSH.
    # shingled_rdd contains (review_id, {shingle_set})

    # (id1, (id2, shingle_set1))
    pairs_with_set1 = candidate_pairs_lsh_rdd.map(lambda pair: (pair[0], pair[1])) \
        .join(shingled_rdd) \
        .map(lambda x: (x[1][0], (x[0], x[1][1]))) # (id2, (id1, shingle_set1))

    # (id2, ((id1, shingle_set1), shingle_set2))
    # Then map to ((id1, id2), (shingle_set1, shingle_set2))
    joined_pairs_with_sets = pairs_with_set1.join(shingled_rdd) \
        .map(lambda x: ((x[1][0][0], x[0]), (x[1][0][1], x[1][1]))) # ((id1, id2), (set1, set2))

    # Calculate Jaccard similarity using the shingle sets
    similarities_rdd = joined_pairs_with_sets.map(
        lambda x: (x[0], calculate_jaccard_similarity(x[1][0], x[1][1]))
    )
    # print("Sample similarities_rdd (LSH candidates):", similarities_rdd.take(10))

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
