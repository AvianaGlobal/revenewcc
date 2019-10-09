from tqdm.auto import tqdm
import pandas as pd


def input_spr(database):
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
    GROUP BY Supplier, datename(YEAR, Invoice_Date)) 
    SELECT COUNT(*) FROM df             
            """
    return countquery, dataquery


def input_nonspr_rollup(filename, clientname, logging):
    input_df = pd.read_csv(filename, encoding='ISO-8859-1', chunksize=1)
    expected_columns = ['Supplier', 'Total_Invoice_Amount', 'Total_Invoice_Count', 'Year']
    inlist = [col in input_df.columns for col in expected_columns]
    if sum(inlist) != len(expected_columns):
        missinglist = [col for col in expected_columns if col not in input_df.columns]
        logging.info(f'The following columns were expected but not found: {missinglist}..')
        raise SystemExit()
    input_df['Client'] = clientname
    return input_df


def input_nonspr_raw(filename2, clientname, logging):
    temp_df = pd.read_csv(filename2, encoding='ISO-8859-1', low_memory=False)
    expected_columns = ['Vendor Name', 'Invoice Date', 'Gross Invoice Amount']
    inlist = [col in temp_df.columns for col in expected_columns]
    if sum(inlist) != len(expected_columns):
        missinglist = [col for col in expected_columns if col not in temp_df.columns]
        logging.info(f'The following columns were expected but not found: {missinglist}..')
        raise SystemExit()
    temp_df = temp_df[expected_columns].rename(columns={'Vendor Name': 'Supplier', })
    temp_df['Client'] = clientname
    temp_df['Year'] = pd.to_datetime(temp_df['Invoice Date']).dt.year
    sums = temp_df.groupby(['Supplier', 'Year'], as_index=False)['Gross Invoice Amount'].sum()
    counts = temp_df.groupby(['Supplier', 'Year'], as_index=False)['Invoice Date'].agg(np.size)
    input_df = pd \
        .merge(sums, counts, on=['Supplier', 'Year']) \
        .rename(columns={'Gross Invoice Amount': 'Total_Invoice_Amount',
                         'Invoice Date': 'Total_Invoice_Count', })
    return input_df


if __name__ == '__main__':
    ''
