import streamlit as st
#from utils import columns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
from sklearn.ensemble import (AdaBoostRegressor, 
                              RandomForestRegressor,
                              VotingRegressor, 
                              GradientBoostingRegressor)
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import (LGBMRegressor,
                      early_stopping)
from sklearn.svm import SVR

from sklearn.base import clone ## sklearn base models for stacked ensemble model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

k_fold = KFold(n_splits = 10, random_state = 11, shuffle = True)



#load model
model = joblib.load(r'C:\Users\Yogitha Hegde\OneDrive\Desktop\models\stacked_model_walmart_98.joblib')



def load_data():
    data = pd.read_csv(r'C:\Users\Yogitha Hegde\OneDrive\Desktop\models\Walmart.csv')
    return data


def display_line_chart(data):
    fig, ax = plt.subplots()
    plt.figure(figsize=(10,40))
    sns.lineplot(data=data,y=data['Weekly_Sales'],x=data['Date'],ax=ax)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("sales trend")
    plt.xticks(rotation=45)
    #plt.tight_layout()
    st.pyplot(fig)

def display_bar_chart(data):
    data_stores = data.nlargest(5,'Weekly_Sales')
    fig, ax = plt.subplots()
    plt.figure(figsize=(10,40))
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data_stores,x=data_stores['Store'],y=data_stores['Weekly_Sales'],ax=ax)
    #sns.boxplot(x='Store', y='Weekly_Sales', data=data,ax=ax)
    plt.xlabel('Store', fontsize=7)
    plt.ylabel('Weekly Sales', fontsize=14)
    plt.title('Weekly Sales by Store', fontsize=16)
    plt.xticks(rotation=45) # Rotate x-axis labels by 45 degrees
    st.pyplot(fig)

def display_heatmap(data):
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(),linewidths=0.5,ax=ax)
    st.pyplot(fig)

def display_box_chart(data):
    fig, ax = plt.subplots()
    plt.figure(figsize=(10,40))
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Holiday_Flag',y='Weekly_Sales',data=data,ax=ax)
    #sns.boxplot(x='Store', y='Weekly_Sales', data=data,ax=ax)
    plt.xlabel('holiday', fontsize=7)
    plt.ylabel('Weekly Sales', fontsize=14)
    plt.title('Weekly Sales and holidays', fontsize=16)
    plt.xticks(rotation=45) # Rotate x-axis labels by 45 degrees
    st.pyplot(fig)

    


    




def main():
    st.title("WALMART SALES PREDICTION")

    data = load_data()
    questions = ['What is the sales trend?','Top 5 stores with highest sales','Display Heatmap','Relation between holidays and sales']
    EDA = st.sidebar.selectbox("INSIGHTS",(questions[0],questions[1],questions[2],questions[3]))

    if EDA == questions[0]:
        display_line_chart(data)
    elif EDA == questions[2]:
        display_heatmap(data)
    elif EDA == questions[3]:
        display_box_chart(data)
    else:
        display_bar_chart(data)

    st.sidebar.title("predict sales")

    holiday_flag = st.sidebar.radio("enter if its a holiday: no=0,yes=1",options=[0,1])
    temperature  = st.sidebar.slider("select temperature",min_value=-2,max_value=100,value=0)
    fuel_price   = st.sidebar.slider("select fule price",min_value=2,max_value=5,value=0)
    cpi = st.sidebar.slider("select CPI",min_value=126,max_value=228,value=0)
    unemployment = st.sidebar.slider("select Unemployment rate",min_value=3,max_value=15,value=0)
    date = st.sidebar.date_input("Choose a date",min_value = datetime.date(2010,1,1))
    month = date.month
    year=date.year
    WeekOfYear = date.isocalendar().week
    storeenc = 7.859814e+05

    def predict(): 
        row = np.array([holiday_flag,temperature,fuel_price,cpi,unemployment,month,year,WeekOfYear,storeenc]) 
        X = pd.DataFrame([row], columns = [holiday_flag,temperature,fuel_price,cpi,unemployment,month,year,WeekOfYear,storeenc])
        prediction = model.predict(X)
        st.write("Weekly sales prediction for the selected parameters:",prediction)

    trigger = st.sidebar.button('Predict', on_click=predict)







if __name__ == '__main__':
    main()

