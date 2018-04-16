# KNN
k: how many numbers you want to find
## Airbnb housing price
```
import pandas as pd

features = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']

dc_listings = pd.read_csv('listings.csv')

dc_listings = dc_listings[features]
print(dc_listings.shape)

dc_listings.head()
```
let k=3
```
import numpy as np

our_acc_value = 3

dc_listings['distance'] = np.abs(dc_listings.accommodates - our_acc_value)
dc_listings.distance.value_counts().sort_index()
```
use .sample to re-arrange the data
```
dc_listings = dc_listings.sample(frac=1,random_state=0)
dc_listings = dc_listings.sort_values('distance')
dc_listings.price.head()
```
in the original table, the price column contains 'string', need to change it first
```
dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float)

mean_price = dc_listings.price.iloc[:5].mean()
mean_price
```
modle evaluation
```
dc_listings.drop('distance',axis=1)

train_df = dc_listings.copy().iloc[:2792]
test_df = dc_listings.copy().iloc[2792:]
```
use a def to do the above thing
```
def predict_price(new_listing_value,feature_column):
    temp_df = train_df
    temp_df['distance'] = np.abs(dc_listings[feature_column] - new_listing_value)
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return(predicted_price)

test_df['predicted_price'] = test_df.accommodates.apply(predict_price,feature_column='accommodates')

test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
mse = test_df['squared_error'].mean()
rmse = mse ** (1/2)
rmse
```
For different features
```
for feature in ['accommodates','bedrooms','bathrooms','number_of_reviews']:
    #test_df['predicted_price'] = test_df.accommodates.apply(predict_price,feature_column=feature)
    test_df['predicted_price'] = test_df[feature].apply(predict_price,feature_column=feature)
    test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
    mse = test_df['squared_error'].mean()
    rmse = mse ** (1/2)
    print("RMSE for the {} column: {}".format(feature,rmse))
```
combine all information we got
data standarization, to decrease diffenrent influence caused by different kind of data--zscore normalization

using ```StandardScaler().fit_transform``` function

```
import pandas as pd
from sklearn.preprocessing import StandardScaler
features = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']

dc_listings = pd.read_csv('listings.csv')

dc_listings = dc_listings[features]

dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float)

dc_listings = dc_listings.dropna()

dc_listings[features] = StandardScaler().fit_transform(dc_listings[features])
normalized_listings = dc_listings

print(dc_listings.shape)

normalized_listings.head()
```
```
norm_train_df = normalized_listings.copy().iloc[0:2792]
norm_test_df = normalized_listings.copy().iloc[2792:]
```
## multi-variable 
use ```distance.euclidean```function to calculate distance from ```from scipy.spatial import distance```
```
from scipy.spatial import distance

first_listing = normalized_listings.iloc[0][['accommodates', 'bathrooms']]
fifth_listing = normalized_listings.iloc[20][['accommodates', 'bathrooms']]
first_fifth_distance = distance.euclidean(first_listing, fifth_listing)
first_fifth_distance


def predict_price_multivariate(new_listing_value,feature_columns):
    temp_df = norm_train_df
    temp_df['distance'] = distance.cdist(temp_df[feature_columns],[new_listing_value[feature_columns]])
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return(predicted_price)

cols = ['accommodates', 'bathrooms']
norm_test_df['predicted_price'] = norm_test_df[cols].apply(predict_price_multivariate,feature_columns=cols,axis=1)    
norm_test_df['squared_error'] = (norm_test_df['predicted_price'] - norm_test_df['price'])**(2)
mse = norm_test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)
```
use sklean to do KNN
```
from sklearn.neighbors import KNeighborsRegressor
cols = ['accommodates','bedrooms']
knn = KNeighborsRegressor()
knn.fit(norm_train_df[cols], norm_train_df['price'])
two_features_predictions = knn.predict(norm_test_df[cols])

from sklearn.metrics import mean_squared_error

two_features_mse = mean_squared_error(norm_test_df['price'], two_features_predictions)
two_features_rmse = two_features_mse ** (1/2)
print(two_features_rmse)
from sklearn.metrics import mean_squared_error
```
by default, n_neighbors=5

more features
```
knn = KNeighborsRegressor()

cols = ['accommodates','bedrooms','bathrooms','beds','minimum_nights','maximum_nights','number_of_reviews']

knn.fit(norm_train_df[cols], norm_train_df['price'])
four_features_predictions = knn.predict(norm_test_df[cols])
four_features_mse = mean_squared_error(norm_test_df['price'], four_features_predictions)
four_features_rmse = four_features_mse ** (1/2)
four_features_rmse
```
