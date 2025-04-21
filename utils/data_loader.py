import pandas as pd
import pycountry

def get_iso3(country_name):
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except:
        return None


def load_data():
    '''
    Load data from various CSV and Excel files --> one combined DataFrame
    Inputs: None
    Returns: one combined DataFrame
    '''
    path_happiness='data/happiness data.xlsx'
    path_lifespan = 'data/lifespan.csv'
    path_gdp = 'data/GDP.csv'
    path_education = 'data/Global_Education.csv'
    path_alcohol = 'data/Alcohol consumption per capita.csv'

    df_happiness = pd.read_excel(path_happiness)
    df_lifespan  = pd.read_csv(path_lifespan)
    df_gdp  = pd.read_csv(path_gdp)
    df_education = pd.read_csv(path_education)
    df_alcohol = pd.read_csv(path_alcohol)
    
    df_happiness = df_happiness[df_happiness['Year'] == 2023].reset_index()

    df_happiness = df_happiness[['Country name','Ladder score']]
    df_lifespan = df_lifespan[['Country Name','2023']]
    df_gdp = df_gdp[['Country Name','2023']]
    df_education = df_education[['Countries and areas','Unemployment_Rate','Birth_Rate']]
    df_alcohol = df_alcohol[['name',' liters of pure alcohol']]

    df_happiness = df_happiness.rename(columns={'Country name':'Country Name','Ladder score':'Happiness Score'})
    df_lifespan = df_lifespan.rename(columns={'2023':'Life Expectancy'})
    df_gdp = df_gdp.rename(columns={'2023':'GDP per Capita'})
    df_education = df_education.rename(columns={'Countries and areas':'Country Name',
                                                'Unemployment_Rate':'Unemployment Rate',
                                                'Birth_Rate':'Birth Rate'})
    df_alcohol = df_alcohol.rename(columns={'name':'Country Name',' liters of pure alcohol':'Alcohol (Liters/Person/Year)'})
    
    df_merged_1 = pd.merge(df_happiness, df_lifespan, on='Country Name')
    df_merged_2 = pd.merge(df_merged_1, df_gdp, on='Country Name')
    df_merged_3 = pd.merge(df_merged_2, df_alcohol, on='Country Name')
    df_merged = pd.merge(df_merged_3, df_education, on='Country Name')

    df_merged['Country'] = df_merged['Country Name'].apply(get_iso3)

    return df_merged

if __name__ == "__main__":
    df = load_data()
    df.to_csv('models/full_data.csv')