#Read the datasets
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

df=pd.read_csv("./datasets/Train.csv")

def convert(df): #Convert the dataframe to a list of dictionaries
    SignDict={"Seat":0,"Enough/Satisfied":1,"Mosque":2,"Temple":3,"Friend":4,"Me":5,"Church":6,"You":7,"Love":8}
    df['Label']=df['Label'].apply(lambda x:SignDict[str(x)])
    return df['Label']
convert(df)
