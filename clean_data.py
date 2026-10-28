import pandas as pd
import os
from langdetect import detect, DetectorFactory, LangDetectException
import argparse

# Keep language detection results reproducible.
DetectorFactory.seed = 0

def is_english(text):
    """Check whether the review text is written in English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

parser = argparse.ArgumentParser(description="Clean Steam reviews from a JSON dataset.")
parser.add_argument("input_file", type=str, help="Path to the input JSON dataset")
parser.add_argument("output_file", type=str, help="Path to the output CSV dataset")
args = parser.parse_args()
        
# Resolve relative file paths from the script's directory.
os.chdir(os.path.dirname(__file__))
file = pd.read_json(args.input_file)
def getplayTime(x):
    return x.get("playtime_at_review")

# Extract playtime at review from each review's author data.
result=[]
for author in file["author"]:
    playtime =getplayTime(author)
    result.append(playtime)
file["playtime_at_review"] = result
# Keep only the columns needed for analysis.
file = file[["recommendationid","review","voted_up","language","timestamp_created","timestamp_updated","playtime_at_review"]].copy()

# Remove duplicate reviews based on review ID.
file = file.drop_duplicates(subset=["recommendationid"])

# Check the number of reviews before language filtering.
print("Before language filtering:")
print(file["voted_up"].value_counts())
print("\n")

# Remove non-English reviews and check the number of reviews after language filtering.
file = file[file["review"].apply(is_english)].copy()
print("After language filtering:")
print(file["voted_up"].value_counts())
print("\n")

# Shuffle the cleaned dataset without changing class counts.
file = file.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the cleaned dataset to the requested output path.
file.to_csv(args.output_file, index=False)
print(file.head(10))
