class DataLoader:
    def __init__(self, wind_turbine, years, features_list, column_names):
        self.wind_turbine = wind_turbine
        self.years = years
        self.features_list = features_list
        self.column_names = column_names

    def load_from_pickle(self, path):
        import pandas as pd
        data = pd.DataFrame(columns=self.column_names)

        for year in self.years:
            data_year = pd.read_pickle(f'{path}/Data/MANF{year}.pkl')

            data_year.columns = data_year.columns.str.replace(r'\(.*?\)', '', regex=True).str.strip()
            data_year = data_year[self.features_list]
            data_year.columns = self.column_names

            data = pd.concat([data, data_year])

        data.reset_index(inplace=True)
        data.rename(columns={'index': 'date'}, inplace=True)
        data = data.sort_values(by='date').reset_index(drop=True)

        return data

            