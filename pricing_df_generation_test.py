import pandas as pd

from src.pricing_df_generation import calculate_column_weight, generate_pricing_df

df_master=pd.read_csv('../data/master/group1.csv',header=0)

# columns_weight_dic={0:1,1:2,2:3,3:3,4:5,5:6}

columns_weight_dic={0:1,1:2}

fd_columns=[0,1]

weights = calculate_column_weight(columns_weight_dic)

print(weights)

df=generate_pricing_df(df_master,weights,fd_columns)
print(df)

price=0

for row in range(0,len(df)):
    for col in range(0,len(df.columns)):
        price=price+df.ix[row,col]

print(price)







