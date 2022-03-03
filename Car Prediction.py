import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder,  StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_log_error,mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

training = pd.read_csv("/home/mwanikii/Documents/Data Science Hackathon/train.csv")
test = pd.read_csv("/home/mwanikii/Documents/Data Science Hackathon/test.csv")
all_data = pd.concat([training, test]) #concataneting data

df = training
#Removing ID
df.drop('ID', axis=1, inplace=True)

#dealing with missing data 
df['Levy'] = df['Levy'].replace('-', np.nan)
df['Levy'] = df['Levy'].astype(np.float64)
Levy_blank = 0
df['Levy'].fillna(Levy_blank, inplace=True)

#Tweaking mileage to numerical data
df['Mileage'] = df['Mileage'].apply(lambda x:x.split(' ')[0])
df['Mileage'] = df['Mileage'].astype(np.int64)

#Tweaking Engine Volume
df['Turbo'] =df['Engine volume'].apply(lambda x:1 if 'Turbo' in str(x) else 0)
df['Engine volume'] = df['Engine volume'].apply(lambda x:str(x).replace('Turbo',''))
df['Engine volume'] = df['Engine volume'].astype(np.float64)
 
#Bringing uniformity to the 'Doors' section
df['Doors'] = df['Doors'].apply(lambda x:str(x).replace('>',''))#Assumes that the vehicle has more than 5 doors
df['Doors'] = df['Doors'].apply(lambda x:str(x).replace('-May',''))
df['Doors'] = df['Doors'].apply(lambda x:str(x).replace('-Mar',''))
df['Doors']= df['Doors'].astype(np.int64)

#Checking for outliers
df_num = ['Doors','Price','Levy','Prod. year','Engine volume','Mileage','Cylinders','Airbags']

#Interquartile Range method can be used to remove outliers in different categories
def find_outliers_limit(df, df_num):
    print(df_num)
    print('-'*50)
    #removing outliers
    q25, q75 = np.percentile(df[df_num], 25),  np.percentile(df[df_num], 75)
    iqr = q75 - q25 #Interquartile Range
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25,q75, iqr))
    #calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q25-cut_off, q75+cut_off
    print('Lower:',lower,'Upper:',upper)
    return lower, upper
def remove_outlier(df, df_num, upper, lower):
    #identify outliers
    outliers = [x for x in df[df_num] if x <= upper]
    print ('Identified outliers: %d' % len(outliers))
    #remove outliers
    outliers_removed = [x for x in df[df_num] if x >=lower and x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))
    final = np.where(df[df_num]>upper, upper, np.where(df[df_num]<lower, lower, df[df_num]))
    return final
df_outlier=['Levy','Engine volume','Mileage','Cylinders','Prod. year']
for df_num in df_outlier:
    lower, upper = find_outliers_limit(df, df_num)
    df[df_num] = remove_outlier(df,df_num,upper,lower)

#plt.figure(figsize=(20,10))
#df[df_outlier].boxplot()
#plt.show()


#Creating new features using binning
labels = [0,1,2,3,4,5,6,7,8,9]
df['Mileage_bin'] = pd.cut(df['Mileage'], len(labels), labels=labels)
df['Mileage_bin'] = df['Mileage_bin'].astype(float)
labels = [0,1,2,3,4]
df['EV_bin'] = pd.cut(df['Engine volume'], len(labels), labels=labels)
df['EV_bin'] = df['EV_bin'].astype(float)

#Handling categorical values through binning
df_cat = df.select_dtypes(include=object)
df_num = df.select_dtypes(include=np.number)
encoding = OrdinalEncoder()
cat_cols = df_cat.columns.tolist()
encoding.fit(df_cat[cat_cols])
cat_oe = encoding.transform(df_cat[cat_cols])
cat_oe = pd.DataFrame(cat_oe, columns=cat_cols)
df_cat.reset_index(inplace=True, drop=True)
print(cat_oe.head())
df_num.reset_index(inplace=True, drop=True)
cat_oe.reset_index(inplace=True, drop=True)
final_all_df=pd.concat([df_num, cat_oe], axis=1)

final_all_df['price_log']=np.log(final_all_df['Price'])
plt.figure(figsize=(20,10))
sns.heatmap(round(final_all_df.corr(),2), annot = True)
#plt.show()

#Performing data scaling and splitting
cols_drop=['Price', 'price_log', 'Cylinders']
X = final_all_df.drop(cols_drop, axis=1)
y = final_all_df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 25)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Training Model 

def train_ml_model(x,y,model_type):

    if model_type == 'lr':
        
        model=LinearRegression()

    elif model_type == 'xgb':

        model=XGBRegressor()
    
    elif model_type == 'rf':

        model=RandomForestRegressor()

    model.fit(X_train_scaled,np.log(y))

    return model

#Model Evaluation
def model_evaluate(model,x,y):

    predictions=model.predict(x)

    predictions=np.exp(predictions)

    mse=mean_squared_error(y,predictions)

    mae=mean_absolute_error(y,predictions)

    mape=mean_absolute_percentage_error(y,predictions)

    msle=mean_squared_log_error(y,predictions)

    mse=round(mse,2)
    mae=round(mae,2)
    mape=round(mape,2)
    msle=round(msle,2)
    
    return [mse,mae,mape,msle]



model_lr = train_ml_model(X_train_scaled,y_train,'lr')
model_xgb = train_ml_model(X_train_scaled,y_train,'xgb')
model_rf = train_ml_model(X_train_scaled,y_train,'rf')

#model_lr_evaluation = model_evaluate(model_lr,X_test,y_test)
model_xgb_evaluation = model_evaluate(model_xgb,X_test,y_test)
model_rf_evaluation = model_evaluate(model_rf,X_test,y_test)

### Deep Learning

### Small Network

model_dl_small=Sequential()

model_dl_small.add(Dense(16,input_dim=X_train_scaled.shape[1],activation='relu'))

model_dl_small.add(Dense(8,activation='relu'))

model_dl_small.add(Dense(4,activation='relu'))

model_dl_small.add(Dense(1,activation='linear'))

model_dl_small.compile(loss='mean_squared_error',optimizer='adam')

model_dl_small.summary()

epochs=20

batch_size=10

model_dl_small.fit(X_train_scaled,np.log(y_train),verbose=0,validation_data=(X_test_scaled,np.log(y_test)),epochs=epochs,batch_size=batch_size)

#plot the loss and validation loss of the dataset

history_df = pd.DataFrame(model_dl_small.history.history)

plt.figure(figsize=(20,10))

plt.plot(history_df['loss'], label='loss')

plt.plot(history_df['val_loss'], label='val_loss')

plt.xticks(np.arange(1,epochs+1,2))

plt.yticks(np.arange(1,max(history_df['loss']),0.5))

plt.legend()

plt.grid()

### Large Network

model_dl_large=Sequential()

model_dl_large.add(Dense(64,input_dim=X_train_scaled.shape[1],activation='relu'))

model_dl_large.add(Dense(32,activation='relu'))

model_dl_large.add(Dense(16,activation='relu'))

model_dl_large.add(Dense(1,activation='linear'))

model_dl_large.compile(loss='mean_squared_error',optimizer='adam')

model_dl_large.summary()

epochs=20

batch_size=10

model_dl_large.fit(X_train_scaled,np.log(y_train),verbose=0,validation_data=(X_test_scaled,np.log(y_test)),epochs=epochs,batch_size=batch_size)

#plot the loss and validation loss of the dataset

history_df = pd.DataFrame(model_dl_large.history.history)

plt.figure(figsize=(20,10))

plt.plot(history_df['loss'], label='loss')

plt.plot(history_df['val_loss'], label='val_loss')

plt.xticks(np.arange(1,epochs+1,2))

plt.yticks(np.arange(1,max(history_df['loss']),0.5))

plt.legend()

plt.grid()

summary=PrettyTable(['Model','MSE','MAE','MAPE','MSLE'])

summary.add_row(['LR']+model_evaluate(model_lr,X_test_scaled,y_test))

summary.add_row(['XGB']+model_evaluate(model_xgb,X_test_scaled,y_test))

summary.add_row(['RF']+model_evaluate(model_rf,X_test_scaled,y_test))

summary.add_row(['DL_SMALL']+model_evaluate(model_dl_small,X_test_scaled,y_test))

summary.add_row(['DL_LARGE']+model_evaluate(model_dl_large,X_test_scaled,y_test))

print(summary)

y_pred=np.exp(model_rf.predict(X_test_scaled))

number_of_observations=20

x_ax = range(len(y_test[:number_of_observations]))

plt.figure(figsize=(20,10))

plt.plot(x_ax, y_test[:number_of_observations], label="True")

plt.plot(x_ax, y_pred[:number_of_observations], label="Predicted")

plt.title("Car Price - True vs Predicted data")

plt.xlabel('Observation Number')

plt.ylabel('Price')

plt.xticks(np.arange(number_of_observations))

plt.legend()

plt.grid()

plt.show()
