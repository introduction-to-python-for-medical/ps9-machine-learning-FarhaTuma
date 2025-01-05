
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
parkinsons = pd.read_csv('parkinsons.csv')

parkinsons_df=parkinsons_df.dropna()
parkinsons_df.head()

parkinsons = pd.read_csv('/content/parkinsons.csv')
sns.pairplot(parkinsons, hue='status')
plt.show()

input_features = ['Shimmer:APQ3', 'RPDE']  
output_feature = ['status']
# Based on the pairplot or other EDA (not shown here for brevity), 
# let's choose 'MDVP:Fo(Hz)' and 'MDVP:Flo(Hz)' as inputs and 'status' as the output

# Define input features (X) and output feature (y)
X = parkinsons_df[['MDVP:Fo(Hz)', 'MDVP:Flo(Hz)']]
y = parkinsons_df['status']

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the input features
X_scaled = scaler.fit_transform(X)

# prompt: Divide the dataset into a training set and a validation set.

from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train.values.ravel())
y_pred = knn.predict(X_val)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

if accuracy < 0.8:
    print("Accuracy is below the required threshold of 0.8. Please adjust the model or features.")

import joblib

joblib.dump(knn, 'my_model.joblib')
