class Defaults:
    dsn = "cc"
    database = "RevenewTest"
    driver = "/usr/local/lib/libmsodbcsql.13.dylib"
    filename = "revenewCC/tests/MissingColumns.csv"
    filename2 = "revenewCC/tests/QEP2019_invoice.csv"
    outputdir = "/Users/mj/Desktop/"
    clientname = 'equinor'
    host = '208.43.250.18'
    port = '51949'
    user = 'sa'
    password = 'Aviana$92821'
    cnxn_str = f'mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver={driver}'


def connectdb(cnxn_str):
    from sqlalchemy import create_engine
    engine = create_engine(cnxn_str, fast_executemany=True, echo=True,)
    return engine.connect()


def readfile(filename, clientname):
    import pandas as pd
    input_df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)
    expected_columns = ['Vendor Name', 'Invoice Date', 'Gross Invoice Amount']
    input_df = input_df[expected_columns]
    input_df['Client'] = clientname
    input_df['Year'] = pd.to_datetime(input_df['Invoice Date']).dt.year
    input_df.groupby('Year').agg({'Supplier': ['size'], 'Gross Invoice Amount': ['sum']})
    return input_df


if __name__ == '__main__':
    ''
