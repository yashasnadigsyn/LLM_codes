import pandas as pd

corona_df = pd.read_csv("Corona_NLP_train.csv", encoding = "ISO-8859-1")
combined_text = '\n'.join(corona_df['OriginalTweet'].astype(str))
file_path = 'combined_text.txt'
with open(file_path, 'w') as f:
    f.write(combined_text)
    
with open(file_path, "r") as f:
    all_text = f.readlines()
    
all_text.remove("\n")

with open(file_path, "w") as f:
    f.writelines(all_text)