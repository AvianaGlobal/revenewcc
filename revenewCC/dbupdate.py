import sqlite3
import pandas as pd
from tqdm import tqdm

# Set up data connection
import revenewCC.dbconnect

engine = revenewCC.dbconnect.dbconnect()
con = revenewCC.dbconnect.sqliteconnect()


def xref_update(supplier_crossref_list):
    supplier_crossref_list.to_sql('crossref', con, index=False, if_exists='replace', schema='Revenew.dbo')
    supplier_crossref_list.to_pickle('revenewCC/inputdata/crossref.pkl')


def comm_update(commodity_list):
    commodity_list.to_sql('commodities', con, index=False, if_exists='replace', schema='Revenew.dbo')
    commodity_list.to_pickle('revenewCC/inputdata/commodities.pkl')


# Queries
xrefquery = "SELECT Supplier, Supplier_ref FROM Revenew.dbo.crossref"
commquery = "SELECT Supplier, Commodity FROM Revenew.dbo.commodities"

# Get total iterations for progress bar
nxref = 153717  # pd.read_sql_query("SELECT COUNT(*) FROM Revenew.dbo.crossref", con=engine).values[0][0]
ncomm = 3352  # pd.read_sql("SELECT COUNT(*) as Count FROM Revenew.dbo.commodities", con=engine).values[0][0]

# Read supplier crossref list
supplier_crossref_list = pd.DataFrame()
chunks = pd.read_sql(xrefquery, engine, chunksize=1)
for chunk in tqdm(chunks, total=nxref, dynamic_ncols=True):
    supplier_crossref_list = pd.concat([supplier_crossref_list, chunk])

# Read commodity list
commodity_list = pd.DataFrame()
chunks = pd.read_sql(commquery, engine, chunksize=1)
for chunk in tqdm(chunks, total=ncomm, dynamic_ncols=False):
    commodity_list = pd.concat([commodity_list, chunk])

supplier_scorecard = pd.read_sql('SELECT * FROM Revenew.dbo.scorecard', engine)
supplier_scorecard.to_sql('scorecard', con=engine, index=False, if_exists='replace', schema='Revenew.dbo')
supplier_scorecard.to_pickle('revenewCC/inputdata/scorecard.pkl')
