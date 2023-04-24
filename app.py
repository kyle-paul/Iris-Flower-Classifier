import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title('Iris Flowers Binary Classifier')
st.write('This app classifies two type of Iris flowers versicolor and virginica using self-built logistic regression')

st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.9, 7.9, 5.5)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 3.8, 3.0)
    petal_width = st.sidebar.slider('Petal Width', 1.0, 2.5, 1.5)
    
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features
    
# tab
tab1, tab2, tab3 = st.tabs(["Train Model", "View Prediction", "Data View"])   
    
with tab1:
    # dataframe
    df = user_input_features()
    st.subheader('User Input parameters')
    st.write(df)

    # plot 1
    df2 = px.data.iris()
    df2 = df2[df2['species'].isin(['versicolor', 'virginica'])]
    fig = px.scatter_3d(df2, x='sepal_width', y='sepal_length', z='petal_width', color='species')
    st.plotly_chart(fig, use_container_width=True)

    # load model
    model = pickle.load(open('model.pickle', 'rb'))
    df = np.array(df)

    # plot 2
    if st.button("Train Model"):
        X = df2[['sepal_width', 'sepal_length', 'petal_width']].values.copy()
        w = model.coef_[0]
        b = model.intercept_[0]

        def x2_function(x0,x1):
            return (-b - w[0]*x0 - w[1]*x1) / w[2]

        x0_range = np.linspace(X[:,0].max(), X[:,0].min(), 10)
        x1_range = np.linspace(X[:,1].max(), X[:,1].min(), 10)

        x0_grid, x1_grid = np.meshgrid(x0_range, x1_range)
        x2_grid = x2_function(x0_grid, x1_grid)

        fig.add_trace(go.Surface(x=x0_grid, y=x1_grid, z=x2_grid))
        st.plotly_chart(fig, use_container_width=True)


with tab2:
    col1, col2, col3 = st.columns(3)
    # predict
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    with col1:
        st.subheader('Class labels')
        st.write(iris.target_names[1:])

    with col2:
        st.subheader('Prediction of Model')
        st.write(iris.target_names[prediction])

    with col3:
        st.subheader('Prediction Probability')
        st.write(prediction_proba)
    
with tab3:
    st.subheader('Class labels')
    st.write(df2)
