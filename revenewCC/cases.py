from tqdm.auto import tqdm
import pandas as pd


def input_spr(database, engine, clientname):
    dataquery = f"""
    SELECT Supplier,
           datename(YEAR, Invoice_Date) AS Year,
           sum(Gross_Invoice_Amount) AS Total_Invoice_Amount,
           count(Invoice_Number) AS Total_Invoice_Count
    FROM (
             SELECT DISTINCT ltrim(rtrim([Vendor Name])) AS Supplier,
                             [Invoice Date] AS Invoice_Date,
                             [Invoice Number] AS Invoice_Number,
                             [Gross Invoice Amount] AS Gross_Invoice_Amount
             FROM {database}.dbo.invoice
             WHERE [Vendor Name] IS NOT NULL
         ) t
    GROUP BY Supplier, datename(YEAR, Invoice_Date)
    ORDER BY Supplier, datename(YEAR, Invoice_Date)             
            """

    countquery = f"""
    WITH df as (SELECT Supplier,
           datename(YEAR, Invoice_Date) AS Year,
           sum(Gross_Invoice_Amount) AS Total_Invoice_Amount,
           count(Invoice_Number) AS Total_Invoice_Count
    FROM (
             SELECT DISTINCT ltrim(rtrim([Vendor Name])) AS Supplier,
                             [Invoice Date] AS Invoice_Date,
                             [Invoice Number] AS Invoice_Number,
                             [Gross Invoice Amount] AS Gross_Invoice_Amount
             FROM {database}.dbo.invoice
             WHERE [Vendor Name] IS NOT NULL
         ) t
    GROUP BY Supplier, datename(YEAR, Invoice_Date)
    ORDER BY Supplier, datename(YEAR, Invoice_Date))
    SELECT COUNT(*) FROM df             
            """
    count = pd.read_sql(countquery, engine)
    input_df = pd.DataFrame()
    chunks = pd.read_sql(dataquery, engine)
    for chunk in tqdm(chunks, total=count):
        input_df = pd.concat([input_df, chunk])
    input_df['Client'] = clientname


if __name__ == '__main__':
    ''
