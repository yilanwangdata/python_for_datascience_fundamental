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
multi-variable 
