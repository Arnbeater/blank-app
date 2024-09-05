import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class DataAnalyseApp:
    def __init__(self):
        self.data = None

    def indlæs_data(self, filsti):
        self.data = pd.read_csv(filsti)
        return "Data indlæst succesfuldt."

    def vis_data_oversigt(self):
        st.write(self.data.describe())
        st.write(self.data.info())

    def visualiser_data(self, x_kolonne, y_kolonne):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x=x_kolonne, y=y_kolonne)
        plt.title(f"{y_kolonne} vs {x_kolonne}")
        st.pyplot(plt)

    def udfør_lineær_regression(self, x_kolonne, y_kolonne):
        X = self.data[[x_kolonne]]
        y = self.data[y_kolonne]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Middelkvadratsafvigelse: {mse}")
        st.write(f"R-kvadrat score: {r2}")

        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', label='Faktiske værdier')
        plt.plot(X_test, y_pred, color='red', label='Forudsigelser')
        plt.title(f"Lineær regression: {y_kolonne} vs {x_kolonne}")
        plt.xlabel(x_kolonne)
        plt.ylabel(y_kolonne)
        plt.legend()
        st.pyplot(plt)

# Streamlit app
st.title('Data Analysis App')

app = DataAnalyseApp()

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    data_load_state = st.text('Loading data...')
    app.indlæs_data(uploaded_file)
    data_load_state.text('Data loaded successfully!')

    if st.button('Show Data Overview'):
        app.vis_data_oversigt()

    x_column = st.selectbox('Select X column for visualization', app.data.columns)
    y_column = st.selectbox('Select Y column for visualization', app.data.columns)
    if st.button('Visualize Data'):
        app.visualiser_data(x_column, y_column)

    if st.button('Perform Linear Regression'):
        app.udfør_lineær_regression(x_column, y_column)
