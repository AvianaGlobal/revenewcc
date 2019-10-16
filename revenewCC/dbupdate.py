import pandas as pd
from tqdm import tqdm
import sys

def dbconnect():
    # Monkey patch for macOS
    if sys.platform == 'darwin':
        host = '208.43.250.18'
        port = '51949'
        user = 'sa'
        password = 'Aviana$92821'
        database = 'AvianaML'
        driver = '/usr/local/lib/libmsodbcsql.13.dylib'
        cnxn_str = f'mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver={driver}'
    else:
        dsn = 'cc'
        cnxn_str = f'mssql+pyodbc://@{dsn}'

    # Make database connection engine
    from sqlalchemy import create_engine

    # Options below are useful for debugging
    engine = create_engine(cnxn_str)
    return engine.connect()


def sqliteconnect():
    import sqlite3
    return sqlite3.connect('revenewCC/inputdata/revenewCC.db')

# Set up data connection
engine = dbconnect()


def xref_update(supplier_crossref_list):
    supplier_crossref_list.to_sql('crossref', con=engine, index=False, if_exists='replace', schema='Revenew.dbo')
    supplier_crossref_list.to_pickle('revenewCC/inputdata/crossref.pkl')


def comm_update(commodity_list):
    commodity_list.to_sql('commodities', con=engine, index=False, if_exists='replace', schema='Revenew.dbo')
    commodity_list.to_pickle('revenewCC/inputdata/commodities.pkl')


def scard_update(supplier_scorecard):
    supplier_scorecard.to_sql('scorecard', con=engine, index=False, if_exists='replace', schema='Revenew.dbo')
    supplier_scorecard.to_pickle('revenewCC/inputdata/scorecard.pkl')


def xref_read():
    supplier_crossref_list = pd.DataFrame()
    xrefquery = "SELECT Supplier, Supplier_ref FROM Revenew.dbo.crossref"
    nxref = pd.read_sql_query("SELECT COUNT(*) FROM Revenew.dbo.crossref", con=engine).values[0][0]
    chunks = pd.read_sql(xrefquery, engine, chunksize=1)
    for chunk in tqdm(chunks, total=nxref, dynamic_ncols=True):
        supplier_crossref_list = pd.concat([supplier_crossref_list, chunk])
    return supplier_crossref_list


def comm_read():
    commodity_list = pd.DataFrame()
    commquery = "SELECT Supplier, Commodity FROM Revenew.dbo.commodities"
    ncomm = pd.read_sql("SELECT COUNT(*) as Count FROM Revenew.dbo.commodities", con=engine).values[0][0]
    chunks = pd.read_sql(commquery, engine, chunksize=1)
    for chunk in tqdm(chunks, total=ncomm, dynamic_ncols=False):
        commodity_list = pd.concat([commodity_list, chunk])
    return commodity_list


def scard_read():
    supplier_scorecard = pd.read_sql('SELECT * FROM Revenew.dbo.scorecard', engine)
    return supplier_scorecard
