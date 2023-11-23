import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## get data raw
df = pd.read_pickle('model_dev_1/data/raw/crime.pkl')

## get column names
df.columns

## do some data cleaning of colun names, 
## make them all lower case, replmove white spaces and rpelace with _ 
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

## get data types
df.dtypes # nice combination of numbers and strings/objects 
len(df)

## drop columns
to_drop = [
    'dr_no',
    'date_rptd',
    'area_',
    'rpt_dist_no',
    'part_1-2'
    'crm_cd',
    'mocodes',
    'premis_cd',
    'weapon_used_cd',
    'status',
    'status_desc',
    'crm_cd_1',
    'crm_cd_2',
    'crm_cd_3',
    'crm_cd_4',
    'location',
    'cross_street',
    'lat',
    'lon'
]
df.drop(to_drop, axis=1, inplace=True, errors='ignore')
df.sample(7)
df.shape
df.columns

## clean date_occ column so it is just the date without the time
df['date_occ'] = pd.to_datetime(df['date_occ']).dt.date
## now encode date_occ so it is a day of the week
df['date_occ'] = pd.to_datetime(df['date_occ'])
df['date_occ'] = df['date_occ'].dt.day_name()

## perform ordinal encoding on date_occ
enc = OrdinalEncoder()
enc.fit(df[['date_occ']])
df['date_occ'] = enc.transform(df[['date_occ']])

## create dataframe with mapping
df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['date_occ'])
df_mapping_date['date_occ_ordinal'] = df_mapping_date.index
df_mapping_date

## save mapping to csv
df_mapping_date.to_csv('model_dev_1/data/processed/mapping_dates.csv', index=False)

## area_name --> will need to encode this
df.area_name.value_counts()
## perform orindal encoding on area_name
enc = OrdinalEncoder()
enc.fit(df[['area_name']])
df['area_name'] = enc.transform(df[['area_name']])

## create dataframe with mapping
df_mapping_area = pd.DataFrame(enc.categories_[0], columns=['area_name'])
df_mapping_area['area_name_ordinal'] = df_mapping_area.index
df_mapping_area.head(10)

# save mapping to csv
df_mapping_area.to_csv('model_dev_1/data/processed/mapping_area_names.csv', index=False)

## perform ordinal encoding on crm_cd_desc
enc = OrdinalEncoder()
enc.fit(df[['crm_cd_desc']])
df['crm_cd_desc'] = enc.transform(df[['crm_cd_desc']])

## create dataframe with mapping
df_mapping_crm = pd.DataFrame(enc.categories_[0], columns=['crm_cd_desc'])
df_mapping_crm['crm_cd_desc_ordinal'] = df_mapping_crm.index
df_mapping_crm.head(7)
## save mapping to csv
df_mapping_crm.to_csv('model_dev_1/data/processed/mapping_crm.csv', index=False)

## perform ordinal encoding on premis_desc
enc = OrdinalEncoder()
enc.fit(df[['premis_desc']])
df['premis_desc'] = enc.transform(df[['premis_desc']])

## create dataframe with mapping
df_mapping_premis = pd.DataFrame(enc.categories_[0], columns=['premis_desc'])
df_mapping_premis['premis_desc_ordinal'] = df_mapping_premis.index
df_mapping_premis.head(5)
df_mapping_premis.to_csv('model_dev_1/data/processed/mapping_premis.csv', index=False)

## get count of missing for weapon_desc
df.weapon_desc.isna().sum()
## replace isna with 'No Weapon'
df.weapon_desc.fillna('Not Reported', inplace=True)
## perform ordinal encoding on weapon_desc
enc = OrdinalEncoder()
enc.fit(df[['weapon_desc']])
df['weapon_desc'] = enc.transform(df[['weapon_desc']])

## create dataframe with mapping
df_mapping_weapon = pd.DataFrame(enc.categories_[0], columns=['weapon_desc'])
df_mapping_weapon['weapon_desc_ordinal'] = df_mapping_weapon.index
df_mapping_weapon.head(10)
# save mapping to csv
df_mapping_weapon.to_csv('model_dev_1/data/processed/mapping_weapons.csv', index=False)

## vict_sex
df.vict_sex.value_counts()
## drop row if sex is equal to X or H
df = df[df['vict_sex'] != 'X' ]
df = df[df['vict_sex'] != 'H' ]
df = df[df['vict_sex'] != 'N']
df = df[df['vict_sex'] != '-']
df.vict_sex.value_counts()


## perform ordinal encoding on vict_sex
enc = OrdinalEncoder()
enc.fit(df[['vict_sex']])
df['vict_sex'] = enc.transform(df[['vict_sex']])
df.vict_sex.value_counts()

## create dataframe with mapping
df_mapping_sex = pd.DataFrame(enc.categories_[0], columns=['vict_sex'])
df_mapping_sex['vict_sex_ordinal'] = df_mapping_sex.index
df_mapping_sex.head(10)
# save mapping to csv
df_mapping_sex.to_csv('model_dev_1/data/processed/mapping_vict_sex.csv', index=False)

len(df)

#### save a temporary csv file of 1000 rows to test the model
df.head(20000).to_csv('model_dev_1/data/processed/crime_20k.csv', index=False)
df