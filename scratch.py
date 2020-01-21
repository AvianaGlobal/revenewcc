import sqlite3

import pandas as pd
from sqlalchemy import create_engine

con = sqlite3.connect('inputdata/cc.db')
username = 'nemesis'
password = 'nemesis'
dsn = 'Revenew'
cnxn_str = f'mssql+pyodbc://{username}:{password}@{dsn}'
engine = create_engine(cnxn_str)
con = engine.connect()

xref_list = pd.read_pickle('crossref.pkl')
cmdty_list = pd.read_pickle('commodity.pkl')
scorecard = pd.read_pickle('scorecard.pkl')
xref_list.to_sql('crossref', con=con, if_exists='replace', index=False)
cmdty_list.to_sql('commodity', con=con, if_exists='replace', index=False)
scorecard.to_sql('scorecard', con=con, if_exists='replace', index=False)

