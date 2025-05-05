
import pandas  as pd
#series example with auto indexing
revenues = pd.Series([25252, 701200, 114980])

revenues.values

revenues.index

type(revenues.values)
#series example with manual indexing

city_revenues = pd.Series(
[4200, 8000, 6500],
index=["Amsterdam", "Toronto", "Tokyo"]
)

city_revenues.values

city_revenues.index

#using loc and .iloc
colors = pd.Series(
["red", "purple", "blue", "green", "yellow"],
    index=[1, 2, 3, 5, 8]
 )
colors

colors.loc[1]
colors.iloc[1]


colors.iloc[1:3]
colors.loc[1:3]



df=pd.read_csv('nba_all_elo.csv')

# identifying the shape of the dataset
df.shape

#checking the top rows of the dataset
df.head(10)


#checking the last rows of the dataset
df.head(10)

#columns
df.columns


#getting to know about the data
df.info()

#basic statistics
stats=df.describe()

import numpy as np
df.describe(include=object)


#exploratory data analysis

df["team_id"].value_counts()



df["date_played"] = pd.to_datetime(df["date_game"])
df.loc[df["team_id"] == "MNL", "date_played"].min()
df.loc[df['team_id'] == 'MNL', 'date_played'].max()
df.loc[df["team_id"] == "MNL", "date_played"].agg(("min", "max"))


df.loc[df["team_id"] == "BOS", "pts"].sum() 


#some querying
df[(df["_iscopy"] == 0) & (df["pts"] > 100) & (df["opp_pts"] > 100) & (df["team_id"] == "BLB")]



df["difference"] = df.pts - df.opp_pts
df["difference"].max()
