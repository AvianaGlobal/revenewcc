dsn = "cc"
database = None
clientname = 'Chemours'
driver = "/usr/local/lib/libmsodbcsql.13.dylib"
filename = "/Users/mj/Downloads/Chemoursrolledup.csv"
# filename2 = "revenewCC/inputdata/TestNonSPR_Raw.csv"
outputdir = "/Users/mj/Desktop/"
host = '208.43.250.18'
port = '51949'
user = 'sa'
password = 'Aviana$92821'
cnxn_str = f'mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver={driver}'
