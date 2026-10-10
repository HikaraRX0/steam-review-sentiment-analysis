import pandas as pd
import os
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

def is_english(text):
    """Check whether the review text is written in English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False
        
# move to current folder,change direction
# _file_ current py path
os.chdir(os.path.dirname(__file__))
file = pd.read_json("reviews_730_raw.json")
def getplayTime(x):
    return x.get("playtime_at_review")

# the playtime is under the author. use applay to get them
result=[]
for author in file["author"]:
    playtime =getplayTime(author)
    result.append(playtime)
file["playtime_at_review"] = result
file = file[["recommendationid","review","voted_up","votes_up","language","timestamp_created","playtime_at_review"]].copy()

# Remove duplicate reviews based on review ID.
file = file.drop_duplicates(subset=["recommendationid"])

# Check the number of reviews before language filtering.
print("Before language filtering:")
print(file["voted_up"].value_counts())

# Remove non-English reviews and check the number of reviews after language filtering.
file = file[file["review"].apply(is_english)].copy()
print("After language filtering:")
print(file["voted_up"].value_counts())

# Separate positive and negative reviews.
positive = file[file["voted_up"] == True]
negative = file[file["voted_up"] == False]

# Randomly undersample both classes to the same size.
n = min(len(positive), len(negative))
file = pd.concat([positive.sample(n=n, random_state=42), negative.sample(n=n, random_state=42)], ignore_index=True)

# Shuffle the balanced dataset.
file = file.sample(frac=1, random_state=42).reset_index(drop=True)

print("After class balancing:")
print(file["voted_up"].value_counts())

file.to_csv("reviews_730.csv", index=False)
print(file.head(10))
