import sqlite3

import pandas as pd
from sqlalchemy import create_engine

dsn = 'Revenew'
cnxn_str = f'mssql+pyodbc://@{dsn}'

engine = create_engine(cnxn_str)
con = engine.connect()
con2 = sqlite3.connect('inputdata/cc.db')

invoice = pd.read_csv('inputdata/TestSPR_Raw.csv', low_memory=False)
xref_list = pd.read_pickle('inputdata/crossref.pkl')
cmdty_list = pd.read_pickle('inputdata/commodity.pkl')
scorecard = pd.read_pickle('inputdata/scorecard.pkl')

invoice.to_sql('invoice', con=con, schema='dbo', if_exists='replace', index=False)
xref_list.to_sql('crossref', con=con, schema='dbo', if_exists='replace', index=False)
cmdty_list.to_sql('commodity', con=con, schema='dbo', if_exists='replace', index=False)
scorecard.to_sql('scorecard', con=con, schema='dbo', if_exists='replace', index=False)
