import kagglehub
import os

# Ensure KAGGLE_USERNAME and KAGGLE_KEY are set in the environment
if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
    print("Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
    print("You can find your API key at https://www.kaggle.com/settings/account")
    exit(1)

try:
    print("Starting dataset download...")
    # Download latest version
    path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
    print(f"Dataset downloaded to: {path}")

    # Create a file to store the path for later steps
    with open("dataset_path.txt", "w") as f:
        f.write(str(path))
    print(f"Dataset path saved to dataset_path.txt")

except Exception as e:
    print(f"An error occurred during download: {e}")
    exit(1)

print("Download script finished.")
