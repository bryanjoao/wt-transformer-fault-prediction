import pandas as pd

if __name__ == '__main__':
    # Split the training set for each wind turbine
    data = pd.read_pickle('./data/train/WT_2019.pkl')

    wt_names_list = ['WT1', 'WT2', 'WT3', 'WT4', 'WT5', 'WT6', 'WT7', 'WT8', 'WT9']

    for wt_name in wt_names_list:
        wt_dataframe = data.filter(like=wt_name)
        wt_dataframe.to_csv(f'./data/train/{wt_name}.csv')


    # Split the test set for each wind turbine
    data = pd.read_pickle('./data/test/WT_2022.pkl')

    wt_names_list = ['WT1', 'WT2', 'WT3', 'WT4', 'WT5', 'WT6', 'WT7', 'WT8', 'WT9']

    for wt_name in wt_names_list:
        wt_dataframe = data.filter(like=wt_name)
        wt_dataframe.to_csv(f'./data/test/{wt_name}.csv')
