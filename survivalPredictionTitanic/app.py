import streamlit as st
import joblib
import pandas as pd

st.sidebar.title('Dashboard')
app_mode=st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

if app_mode=="Home":
    st.header("Titanic Survival Prediction")
    image_path = "survivalPredictionTitanic/beautiful-shot-olympia-shipwreck-amorgos-island-greece_181624-18615.jpg"
    st.image(image_path, use_column_width=True)
    
    st.markdown("""
  ### Titanic Survival Prediction Model 🚢⚓

- Data Preprocessing: Cleaning and transforming the Titanic dataset for analysis.
- Feature Engineering: Extracting and selecting key features to improve model accuracy.
- Model Training: Using [model name(s), e.g., Logistic Regression, Random Forest, etc.] to predict survival outcomes.
- Evaluation: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.
- Visualization: Including insights and visual representations of data trends and model predictions.
### Features:
- Jupyter Notebook with detailed code and explanations.
- Results and insights from the analysis.
- Ready-to-use scripts for replication or further exploration.
- This project is perfect for beginners exploring machine learning or those interested in classic datasets. 🧑‍💻📊

    ### About Us
    Learn more about this project and the team behind it on the **About** page.
    """)

elif app_mode =="About Project" :
     st.header("About")
     st.markdown("""
    ### Titanic Dataset Overview

    The Titanic dataset contains 891 entries with 12 columns, including both numerical and categorical data. Below is an overview:

    - **PassengerId**: Unique identifier for each passenger.
    - **Survived**: Indicates survival (1 = survived, 0 = did not survive).
    - **Pclass**: Passenger's class (1 = First, 2 = Second, 3 = Third).
    - **Name**: Full name of the passenger.
    - **Sex**: Gender of the passenger (male/female).
    - **Age**: Age in years (missing values are NaN).
    - **SibSp**: Number of siblings or spouses aboard.
    - **Parch**: Number of parents or children aboard.
    - **Ticket**: Ticket number.
    - **Fare**: Passenger fare.
    - **Cabin**: Cabin number (most values are missing).
    - **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

    ### Data Insights

    - The dataset has missing values in `Age`, `Cabin`, and `Embarked`.
    - The majority of passengers are male (577 out of 891).
    - The dataset is imbalanced, with approximately 38.4% survival rate.

    ### Preprocessing Steps

    - Handle Missing Values: Impute missing `Age`, `Embarked`, and potentially drop or impute `Cabin`.
    - Feature Encoding: Convert `Sex` and `Embarked` into numerical values for model compatibility.
    - Feature Selection: Use columns like `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked` for prediction.

    This dataset is ideal for binary classification tasks and serves as a classic example of predictive modeling.
    

    """)
     
elif app_mode=="Prediction":
      model = joblib.load(r"D:\target\ml\survivalPredictionTitanic\Titanic.joblib")
      st.header("Survival Prediction")

      st.write("Please fill in the details below:")

      # Input fields
      pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
      sex = st.selectbox("Sex", ["male", "female"])
      age = st.number_input("Age", min_value=0, max_value=100, value=30)
      sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
      parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
      fare = st.number_input("Fare", min_value=0.0, value=50.0)
      embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

      # Preprocess inputs
      sex_encoded = 1 if sex == "male" else 0
      embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]

      # Predict button
      if st.button("Predict Survival"):
          input_data = pd.DataFrame({
              "Pclass": [pclass],
              "Sex": [sex_encoded],
              "Age": [age],
              "SibSp": [sibsp],
              "Parch": [parch],
              "Fare": [fare],
              "Embarked": [embarked_encoded]
          })

          prediction = model.predict(input_data)[0]
          st.snow()
          if prediction == 1:
              st.success("The passenger is likely to survive.")
          else:
              st.error("The passenger is unlikely to survive.")
