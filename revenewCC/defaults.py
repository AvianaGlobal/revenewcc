dsn = "cc"
user = 'sa'
password = 'Aviana$92821'
cnxn_dsn = f'mssql+pyodbc://{user}:{password}@{dsn}'
database = None
clientname = 'Casey'

driver = "/usr/local/lib/libmsodbcsql.13.dylib"
host = '208.43.250.18'
port = '51949'
cnxn_str = f'mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver={driver}'

# filename = "/Users/mj/Downloads/Chemoursrolledup.csv"
filename = 'revenewCC/inputdata/TestNonSPR_Rolledup.csv'
# filename2 = "revenewCC/inputdata/TestNonSPR_Raw.csv"
outputdir = "/Users/mj/Desktop/"

