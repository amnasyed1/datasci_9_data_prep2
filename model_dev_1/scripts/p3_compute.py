import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Import the clean random sample of 10k data
df = pd.read_csv('model_dev_1/data/processed/crime_20k.csv')
df
len(df)

# drop rows with missing values
df.dropna(inplace=True)
len(df)

# Define the features and the target variable
X = df.drop('vict_sex', axis=1)  # Features (all columns except 'arrest')
y = df['vict_sex']               # Target variable (arrest)

# Initialize the StandardScaler
scaler = StandardScaler()
scaler.fit(X) # Fit the scaler to the features
pickle.dump(scaler, open('model_dev_1/models/scaler_crime20k.sav', 'wb')) # Save the scaler for later use

# Fit the scaler to the features and transform
X_scaled = scaler.transform(X)

# Split the scaled data into training, validation, and testing sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check the size of each set
(X_train.shape, X_val.shape, X_test.shape)

# Pkle the X_train for later use in explanation
pickle.dump(X_train, open('model_dev_1/models/X_train_15k.sav', 'wb'))
# Pkle X.columns for later use in explanation
pickle.dump(X.columns, open('model_dev_1/models/X_columns_15k.sav', 'wb'))
