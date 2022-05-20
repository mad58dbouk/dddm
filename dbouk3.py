# importing libraries

import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.express as px
import plotly.graph_objects as go

#use page's entire width 
st.set_page_config(layout="wide")

df1 = pd.read_csv('CLV_new.csv')

#creating menu button

boxx = st.selectbox('Browse through',('Home Page','EDA Report', 'Insightful Visualizations', 'prediction tool'))
if boxx == "Home Page":

    #aautomatic dataset upload
    uploaded_data = st.sidebar.file_uploader('Upload dataset', type='csv')





    


    

    

    if uploaded_data:
        st.header('A look into the Data')
        df1= pd.read_csv(uploaded_data)
        Kpi1, Kpi2, Kpi3, Kpi4 =st.columns(4)

        col = df1.columns

        #creating KPI metrics

        Kpi1.metric(label ="Number of Columns", value = len(col))

        Kpi2.metric(label="Number of rows", value = len(df1))

        Kpi3.metric(label="Null values", value = df1.isnull().sum().sum())

        Kpi4.metric(label="Duplicates", value =df1.duplicated().sum().sum())

        
        

        


        
        st.sidebar.header("filter through")
        employmentStatus=st.sidebar.multiselect("select Employment Status:",options=df1["EmploymentStatus"].unique(),default= df1["EmploymentStatus"].unique())
        state=st.sidebar.multiselect("select State:",options=df1["State"].unique(),default= df1["State"].unique())
        coverage=st.sidebar.multiselect("select Coverage type:",options=df1["Coverage"].unique(),default= df1["Coverage"].unique())

        df1_selection = df1.query("EmploymentStatus == @employmentStatus & State == @state & Coverage ==@coverage")

        st.dataframe(df1_selection)



if boxx == "EDA Report":


    uploaded_data = st.sidebar.file_uploader('Upload dataset', type='csv')





    


    

    

    if uploaded_data:
        
        df1= pd.read_csv(uploaded_data)
        

        Kpi1, Kpi2, Kpi3, Kpi4 =st.columns(4)

    

        Kpi1.metric(label ="Policies", value = df1["Policy_Type"].nunique())

        Kpi2.metric(label="Coverages", value = df1["Coverage"].nunique())

        Kpi3.metric(label="Offer types", value = df1["Renew_Offer_Type"].nunique())

        Kpi4.metric(label="Sales Channels", value =df1["Sales_Channel"].nunique())


        st.write("""Prediction Power exploration""")

        col1, col2, col3 = st.columns(3)

        df_employmentstatus = df1.groupby(["EmploymentStatus"])["Customer"].count().reset_index(name="counts")


        fig8 = px.histogram(df_employmentstatus,x="EmploymentStatus", y= "counts", color = "EmploymentStatus", title="Total number of customers by education")
        fig8=go.Figure(fig8)
        fig8.show(renderer="colab") 

        col1.plotly_chart(fig8, use_container_width = True)

        df_maritalstatus = df1.groupby(["Marital_Status"])["Customer_Lifetime_Value"].mean().reset_index(name="average")

        fig9 = px.histogram(df_maritalstatus,x="Marital_Status", y= "average", color = "Marital_Status", title="Average Customer Lifetime Value per Marital Status")
        go.Figure(fig9)
        fig9.show(renderer= "colab")

        col2.plotly_chart(fig9, use_container_width = True)

        df_complaints = df1.groupby(["Number_of_Open_Complaints"])["Customer_Lifetime_Value"].mean().reset_index(name="average")
        fig12 = px.bar(df_complaints, x = "Number_of_Open_Complaints", y="average", color = "Number_of_Open_Complaints", title = "average Customer Lifetime Versus complaints")
        go.Figure(fig12)
        fig12.show(renderer = "colab")
        col3.plotly_chart(fig12, use_container_width = True)













    
if boxx == "Insightful Visualizations":

    
    uploaded_data = st.sidebar.file_uploader('Upload dataset', type='csv')

    if uploaded_data:
        
        df1= pd.read_csv(uploaded_data)
        Kpi1, Kpi2, Kpi3, Kpi4 =st.columns(4)

        st.header("""This page demonstrates figures of relevance to our main Subject CLV""")

        Kpi1, Kpi2, Kpi3, Kpi4 =st.columns(4)


        Kpi1.metric(label ="Total customers", value = df1['Customer'].count())

        Kpi2.metric(label="Average Customer Lifetime Value $", value = df1['Customer_Lifetime_Value'].mean().round(0))

        Kpi3.metric(label="Average Customer Income $", value = df1['Income'].mean().round(0))

        Kpi4.metric(label="Policies", value = 6)

        



    col1, col2 = st.columns(2)

    
    fig5 = px.treemap(df1, path = ["Location_Code","State"],values = "Customer_Lifetime_Value", hover_name = "State",color = "Customer_Lifetime_Value", height = 500, width= 950, title="Customer lifetime value dominance per regions")
    go.Figure(fig5)
    fig5.show(renderer="colab")

    
    col1.plotly_chart(fig5, use_container_width =True)

    

    with col2:
        df_1 = df1.groupby(["State","Gender"])["Customer"].count().reset_index(name="counts")
        fig7 =px.sunburst(df_1,path = ["Gender","State"],values = "counts", hover_name = "State",color = "counts", height = 490, width=490, title="Customers spread over regions")
        go.Figure(fig7)
        fig7.show(renderer="colab")

        st.plotly_chart(fig7)




    







if boxx == "prediction tool":


    st.write("""please input the corresponding numerics to evaluate the customer's Lifetime value""")

    q1= df1.quantile(0.25)
    q3= df1.quantile(0.75)

    iqr = q3-q1
    df2= df1[~((df1 < (q1-1.5*iqr))|(df1 > (q3+1.5*iqr))).any(axis=1)]

    X2 = df2.drop(columns=["Customer_Lifetime_Value",'Gender', 'Policy_Type', 'Sales_Channel', 'Vehicle_Class', 'Renew_Offer_Type', 'Education', 'Vehicle_Size','Response' ,'Location_Code', 'Marital_Status', 'Coverage', 'Customer', 'Effective_To_Date', 'EmploymentStatus', 'State', 'Policy'], axis=1)

    y2= df2[["Customer_Lifetime_Value"]]
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2,y2, test_size=0.2, random_state=111, shuffle = True)

 


    #random forest:
    from sklearn.ensemble import RandomForestRegressor

    forest_reg = RandomForestRegressor()
    forest_reg.fit(X_train_2, y_train_2)
    


    def user_report():
        
    

        
            
        
        
        
        
        Income= st.sidebar.slider('Income',min_value =0,max_value=100000,step =500)
        
        Monthly_Premium_Auto=st.sidebar.slider('Monthly_Premium_Auto', min_value=0, max_value=298, step=1)
        Months_Since_Last_Claim=st.sidebar.slider('Months since last claim',min_value =0,max_value = 40,step =1)
        Months_Since_Policy_Inception =st.sidebar.slider('Months since last claim',min_value =0,max_value =50,step =1)
        Number_of_Open_Complaints=st.sidebar.slider('Number of Complaints', min_value =0, max_value =  5, step =1)
        Number_of_Policies= st.sidebar.slider('Number of policies',min_value=0,max_value =9,step =1)
        
        
        
        
        Total_Claim_Amount=st.sidebar.slider('total claim amount',min_value =0,max_value= 3000,step=500)
        
        


        
        user_report_data = {
        
        
        
        
        
        
        'Income': Income,
        
        
        'Monthly_Premium_Auto': Monthly_Premium_Auto,
        'Months_Since_Last_Claim': Months_Since_Last_Claim,
        'Months_Since_Policy_Inception': Months_Since_Policy_Inception,
        'Number_of_Open_Complaints':Number_of_Open_Complaints,
        'Number_of_Policies':Number_of_Policies,
        
        
        
        
        'Total_Claim_Amount': Total_Claim_Amount,
        
        

        }

        report_data =pd.DataFrame(user_report_data, index = [0])
        return report_data
    
    
    
    user_data = user_report()
    st.subheader('Customer Lifetime Value')
    st.write(user_data)


    CLV =forest_reg.predict(user_data)
    st.subheader('Customer Lifetime Value')
    st.subheader(CLV)
    st.header("""Above is your customer's expected lifetime value with high accuracy""")
