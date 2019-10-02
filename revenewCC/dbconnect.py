import sys

# Work around macOS issues with ODBC
if sys.platform == 'darwin':
    host = '208.43.250.18'
    port = '51949'
    user = 'sa'
    password = 'Aviana$92821'
    database = 'AvianaML'
    driver = '/usr/local/lib/libmsodbcsql.13.dylib'
    runtype = 'Mac'
    dsn = 'cc'

else:
    runtype = 'Windows'
    dsn = 'cc'


def dbconnect():
    # Monkey patch for macOS
    if runtype == 'Mac':
        cnxn_str = f'mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver={driver}'
    else:
        cnxn_str = f'mssql+pyodbc://@{dsn}'

    # Make database connection engine
    from sqlalchemy import create_engine

    # Options below are useful for debugging
    engine = create_engine(
        cnxn_str,
        fast_executemany=True,
        echo=False,
        # implicit_returning=False,
        # isolation_level="AUTOCOMMIT",
    )
    return engine.connect()


def sqliteconnect():
    import sqlite3
    return sqlite3.connect('revenewCC/inputdata/revenewCC.db')
