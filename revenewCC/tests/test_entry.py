import unittest
import pandas as pd
from sqlalchemy import create_engine


def input_suppliers():
    host = '208.43.250.18'
    port = '51949'
    user = 'sa'
    password = 'Aviana$92821'
    database = 'AvianaML'
    driver = '/usr/local/lib/libmsodbcsql.13.dylib'
    cnxnstr = f'mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver={driver}'
    engine = create_engine(cnxnstr)
    query = '''
        SELECT DISTINCT ltrim(rtrim(lower([Vendor Name]))) AS Supplier
        FROM AvianaML.dbo.invoice i
        ORDER BY ltrim(rtrim(lower([Vendor Name])))
    '''
    suppliers = pd.read_sql(query, engine).dropna()
    return suppliers


class TestInputs(unittest.TestCase):
    def test_columns(self):
        cols = input_suppliers().columns
        asserts = [self.assertIn(container=cols, member=col) for col in cols]
        return asserts


if __name__ == '__main__':
    unittest.main()
