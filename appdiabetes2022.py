from flask import Flask, render_template, request
import joblib
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the joblib file
model_file_path = os.path.join(script_dir, "NewModelDiabetes2022.joblib")
scaler_file_path = os.path.join(script_dir, "Diabetes2022Scaler.joblib")
# Load the model
model = joblib.load(model_file_path)

model_filename = "NewModelDiabetes2022.joblib"
scaler_name = "Diabetes2022Scaler.joblib"
# model_directory = r'C:\Users\User\Desktop\SeniorProjectTest\LifeFactors-master\joblib test'
# full_path = os.path.join(model_directory, model_filename)
# scalerpath = os.path.join(model_directory, scaler_name)
scaler1 = joblib.load(scaler_file_path)

from app import app


# Define the prediction route
@app.route("/predictprediabetes", methods=["POST"])
def predict_prediabetes():
    # Get user input from the form
    Stroke = int(request.form["Stroke"])
    DEnum = int(request.form["DepressionEpisodes"])
    if DEnum == 0: 
        DepressionEpisodes = 1
    else:
        DepressionEpisodes = DEnum
    DiffWalk = int(request.form["DiffWalk"])
    Race = float(request.form["Race"])
    Sex = int(request.form["Sex"])
    age_discreet = int(request.form["age"])
    SmokeStatus = int(request.form["SmokerStatus"])
    VapeStatus = int(request.form["VapeStatus"])
    BingeDrinker = int(request.form["BingeDrinker"])
    DrinksPerWeek = int(request.form["DrinkPerWeek"])
    HealthRating = int(request.form["HealthRating"])
    PhysicalHealth = int(request.form["PhysicalHealth"])
    MentalHealth = int(request.form["MentalHealth"])
    PhysAct = int(request.form["PhysAct"])
    age = 0  # Initialize age variable
    if 18 <= age_discreet <= 24:
        age = 1
    elif 25 <= age_discreet <= 29:
        age = 2
    elif 30 <= age_discreet <= 34:
        age = 3
    elif 35 <= age_discreet <= 39:
        age = 4
    elif 40 <= age_discreet <= 44:
        age = 5
    elif 45 <= age_discreet <= 49:
        age = 6
    elif 50 <= age_discreet <= 54:
        age = 7
    elif 55 <= age_discreet <= 59:
        age = 8
    elif 60 <= age_discreet <= 64:
        age = 9
    elif 65 <= age_discreet <= 69:
        age = 10
    elif 70 <= age_discreet <= 74:
        age = 11
    elif 75 <= age_discreet <= 79:
        age = 12
    else:
        age = 13
    Height = float(request.form["Height"])
    Weight = float(request.form["Weight"])
    heightInM = Height/100
    BMI = Weight/((heightInM)**2)
    
    user_input = {'Stroke':Stroke, 'DepressionEpisodes':DepressionEpisodes, 'DiffWalk':DiffWalk,
                  'Race':Race, 'Sex':Sex, 'AgeCategory':age, 'BMI':BMI,
                  'SmokeStatus':SmokeStatus, 'VapeStatus':VapeStatus, 
                  'BingeDrinker':BingeDrinker, 'DrinkPerWeek':DrinksPerWeek,
                  'HealthRating':HealthRating, 'PhysicalHealth':PhysicalHealth,
                  'MentalHealth':MentalHealth, 'PhysAct':PhysAct}
    
    # """import pandas as pd
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    
    # df2022 = pd.read_csv(r'C:\Users\User\Desktop\SP1\sp1 latest2\brfss2022.csv')
    # df2021 = pd.read_csv(r'C:\Users\User\Desktop\SP1\sp1 latest2\LLCP2021.csv')
    # projectcol2022 = ['CVDSTRK3', 
    #           'ADDEPEV3', 'DIFFWALK','_MRACE2', 
    #           '_SEX', '_AGEG5YR', '_BMI5',
    #           '_SMOKER3','_CURECI2','_RFBING6',
    #           '_DRNKWK2','_RFHLTH','_PHYS14D', 
    #           '_MENT14D','_TOTINDA',
    #           'PREDIAB2',]
    # projectcol2021 = ['CVDSTRK3', 
    #           'ADDEPEV3', 'DIFFWALK','_MRACE1', 
    #           '_SEX', '_AGEG5YR', '_BMI5',
    #           '_SMOKER3','_CURECI1','_RFBING5',
    #           '_DRNKWK1','_RFHLTH','_PHYS14D', 
    #           '_MENT14D','_TOTINDA',
    #           'PREDIAB1']
    # column_mapping2021 = {
    #     'CVDSTRK3':'Stroke', #1 1 Yes, 2 No, 7:Dont Know, 9:Refused
    #     'ADDEPEV3':'DepressionEpisodes', #1 Yes, 2 No, 7 Dont Know, 9 Refused 
    #     'DIFFWALK':'DiffWalk', #2 1 Yes, 2 No, 7, Dont Know, 9 Refused
    #     '_MRACE1': 'Race', #3 1 White, 2 Black or African American Only, 3 American Indian or Alaskan Native, 4 Asian, 5 Multiracial, 6 Multiracial, 88 Others, 77 dont Know, 99 refused 
    #     '_SEX': 'Sex', #4 1 Male, 2 Female
    #     '_AGEG5YR':'AgeCategory', #5 
    #     '_BMI5':'BMI', #6
    #     '_SMOKER3': 'SmokeStatus', #7 1 Current Smoker, 2 Smoke Some days, 3 Former Smoker, 4 Never Smoked, 9 Dont know/Refused
    #     '_CURECI1':'VapeStatus', #8 1 Not use Vape, 2 Use Vape, 9 refused
    #     '_RFBING5': 'BingeDrinker', #9 1 No, 2 Yes (more than 5 drinks for men, More than 4 for female) 
    #     '_DRNKWK1':'DrinkPerWeek', #10 0 No, 1-9899 drinks per week,
    #     '_RFHLTH':'HealthRating', #11 1 Good health, 2 Poor Health, 9 refused
    #     '_PHYS14D':'PhysicalHealth', #12 1 30 days bad phys health, 2 1-13 days bad phys health, 3 14+ days bad phys health, 9 Dont know 
    #     '_MENT14D': 'MentalHealth', #13 1 30 days bad Mental health, 2 1-13 days bad mental health, 3 14+ days bad mental health, 9 Dont know
    #     '_TOTINDA': 'PhysAct', #14 1 Yes, 2 No
    #     'PREDIAB1': 'Pre_Diabetic' #1 Yes, 2 Yes, 3 No, 7 dont know, 9 refused}
    # }

    # column_mapping2022 = {
    #     'CVDSTRK3':'Stroke', #1 Yes, 2 No, 7:Dont Know, 9:Refused
    #     'ADDEPEV3':'DepressionEpisodes', #1 Yes, 2 No, 7 Dont Know, 9 Refused 
    #     'DIFFWALK':'DiffWalk', #1 Yes, 2 No, 7, Dont Know, 9 Refused
    #     '_MRACE2': 'Race', #1 White, 2 Black or African American Only, 3 American Indian or Alaskan Native, 4 Asian, 5 Multiracial, 6 Multiracial, 88 Others, 77 dont Know, 99 refused 
    #     '_SEX': 'Sex', # 1 Male, 2 Female
    #     '_AGEG5YR':'AgeCategory',
    #     '_BMI5':'BMI',
    #     '_SMOKER3': 'SmokeStatus', # 1 Current Smoker, 2 Smoke Some days, 3 Former Smoker, 4 Never Smoked, 9 Dont know/Refused
    #     '_CURECI2':'VapeStatus', #1 Not use Vape, 2 Use Vape, 9 refused
    #     '_RFBING6': 'BingeDrinker', #1 No, 2 Yes (more than 5 drinks for men, More than 4 for female) 
    #     '_DRNKWK2':'DrinkPerWeek', # 0 No, 1-9899 drinks per week,
    #     '_RFHLTH':'HealthRating', #1 Good health, 2 Poor Health, 9 refused
    #     '_PHYS14D':'PhysicalHealth', #1 30 days bad phys health, 2 1-13 days bad phys health, 3 14+ days bad phys health, 9 Dont know 
    #     '_MENT14D': 'MentalHealth', #1 30 days bad Mental health, 2 1-13 days bad mental health, 3 14+ days bad mental health, 9 Dont know
    #     '_TOTINDA': 'PhysAct', #1 Yes, 2 No
    #     'PREDIAB2': 'Pre_Diabetic' #1 Yes, 2 Yes, 3 No, 7 dont know, 9 refused
    # }
    # df2021 = df2021[projectcol2021]
    # df2021.rename(columns=column_mapping2021, inplace=True)
    # df2022 = df2022[projectcol2022]
    # df2022.rename(columns=column_mapping2022, inplace=True)
    # df2021 = df2021.dropna()
    # df2022 = df2022.dropna()


    # print(df2021.shape)
    # print(df2022.shape)
    # finalDF = pd.concat([df2021, df2022], ignore_index=True)
    # print(finalDF.isnull().sum())
    # print(finalDF.columns)
    # print(finalDF.shape)
    # columns_to_check = ['Stroke','DepressionEpisodes', 'DiffWalk', 'Race',
    #                     'SmokeStatus','VapeStatus', 'BingeDrinker', 'HealthRating',
    #                     'PhysicalHealth', 'MentalHealth', 'PhysAct', 'Pre_Diabetic']

    # rows_to_drop = finalDF[columns_to_check].apply(lambda col: (col == 7) | (col == 9) | (col == 88) | (col == 77) | (col == 99)).any(axis=1)
    # print()
    # print()
    # finalDF = finalDF[~rows_to_drop]
    # print(finalDF.shape)
    # df = finalDF
    # df['Pre_Diabetic'] = df['Pre_Diabetic'].map({1: 1, 2: 1, 3: 0})

    # seendata = finalDF.head(150000)
    
    # import pandas as pd
    # # Load the dataset (assuming 'data' contains your dataset)
    # # Replace 'data.csv' with your actual file name or path
    
    # df1=seendata
    
    # df_0 = df1[df1['Pre_Diabetic'] == 0].sample(n=8000, random_state=42, replace=True)
    # df_1 = df1[df1['Pre_Diabetic'] == 1].sample(n=12000, random_state=42, replace= True)
    # df1 = pd.concat([df_0,df_1], ignore_index = True)
        
    # df1 = df1.drop('Pre_Diabetic', axis=1)
    # df1= scaler.fit_transform(df1)
    # """
    scaled_input = scaler1.transform([list(user_input.values())])
    user_input = scaled_input

    # Make prediction using the model
    # prediction = model.predict(np.array([user_input]))
    # probability = model.predict_proba(np.array([user_input]))[:,1]
    
    prediction = model.predict(user_input.reshape(1, -1))
    probability = model.predict_proba(user_input.reshape(1, -1))[:,1]
    
    
   # Interpret the prediction (0: no diabetes, 1: diabetes)
    if prediction == 0:
        result = "Negative. The model predicts you MIGHT NOT BE PREDIABETIC with an accuracy of 66.6%"
        result2 = "The model predicts that your lifestyle is correlated with pre-diabetes with a probability of: "
    else:
         result = "Positive. The model predicts you MIGHT BE PREDIABETIC with an accuracy of 66.6%"
         result2 = "The model predicts that your lifestyle is correlated with pre-diabetes with a probability of: "

    prob = round(probability[0]*100,2)
    


     # Render the prediction page with the result
    return render_template("prediction.html", result=result, result2=result2, probability=prob)


if __name__ == "__main__":
    app.run(debug=True, port=8000)
