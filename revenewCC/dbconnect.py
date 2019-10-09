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
