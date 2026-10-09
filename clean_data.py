import pandas as pd
import os
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
file = file[["recommendationid","review","voted_up","language","timestamp_created","playtime_at_review"]].copy()

# delete the same comment id, not just same review
file = file.drop_duplicates(subset=["recommendationid","review"])
file.to_csv("reviews_730.csv", index=False)
print(file.head(10))
