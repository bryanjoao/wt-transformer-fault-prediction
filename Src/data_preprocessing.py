import pandas as pd

if __name__ == '__main__':
    # Split the training set for each wind turbine
    data = pd.read_pickle('./data/train/MANF_2019.pkl')

    wt_names_list = ['MF02', 'MF03', 'MF04', 'MF05', 'MF06', 'MF07', 'MF09', 'MF12', 'MF13']

    for wt_name in wt_names_list:
        wt_dataframe = data.filter(like=wt_name)
        wt_dataframe.to_csv(f'./data/train/{wt_name}.csv')


    # Split the test set for each wind turbine
    data = pd.read_pickle('./data/test/MANF_2022.pkl')

    wt_names_list = ['MF02', 'MF03', 'MF04', 'MF05', 'MF06', 'MF07', 'MF09', 'MF12', 'MF13']

    for wt_name in wt_names_list:
        wt_dataframe = data.filter(like=wt_name)
        wt_dataframe.to_csv(f'./data/test/{wt_name}.csv')