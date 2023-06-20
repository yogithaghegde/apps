import streamlit as st
#from utils import columns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def load_data():
    data = pd.read_csv(r'Walmart.csv')
    corrmatrix = data.drop(columns=['Date'],inplace=True).corr()
    
    return data,corrmatrix


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
    st.write(data)
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






if __name__ == '__main__':
    main()

