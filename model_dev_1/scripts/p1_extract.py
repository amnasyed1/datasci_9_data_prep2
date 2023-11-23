import pandas as pd 

## get data 

# original link: https://catalog.data.gov/dataset/crime-data-from-2010-to-2019/resource/7019ef5a-a383-479c-8a28-8175ced9b7f5
# data download link: 
datalink = 'https://data.lacity.org/api/views/63jg-8b9z/rows.csv?accessType=DOWNLOAD'

df = pd.read_csv(datalink)
df
df.size
df.sample(5)
df.columns

## save as csv to WK9/code/model_dev/data/raw
df.to_csv('model_dev_1/data/raw/crime.csv', index=False)

## save as pickle to WK9/code/model_dev/data/raw
df.to_pickle('model_dev_1/data/raw/crime.pkl')