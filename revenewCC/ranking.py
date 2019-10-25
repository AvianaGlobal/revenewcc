#!/usr/bin/env python
from gooey import Gooey


@Gooey(program_name='\nRevenewML\nCC Supplier Ranking\n', default_size=(700, 700), image_dir='::gooey/default',
       language_dir='gooey/languages', )
def main():
    # Parse User inputs
    from revenewCC.argparser import parser
    args = parser.parse_args()
    dsn = args.dsn
    clientname = args.clientname
    database = args.database
    outputdir = args.outputdir
    filename = args.filename
    filename2 = args.filename2

    # Import packages
    import os
    import sys
    import time
    import logging
    import numpy as np
    import pandas as pd
    from fuzzywuzzy import fuzz
    from sqlalchemy import create_engine
    from timeit import default_timer as timer
    from revenewCC import helpers

    # Set default threshold for soft-matching
    threshold = 89

    # Make database connection engine (with debug settings)
    cnxn_str = f'mssql+pyodbc://@{dsn}'
    if sys.platform == 'darwin':
        if os.environ['USER'] == 'mj':
            user = 'sa'
            password = 'Aviana$92821'
            cnxn_str = f'mssql+pyodbc://{user}:{password}@{dsn}'
    if sys.platform == 'win32':
        if os.environ['USERNAME'] == 'mj':
            user = 'sa'
            password = 'Aviana$92821'
            cnxn_str = f'mssql+pyodbc://{user}:{password}@{dsn}'
    engine = create_engine(cnxn_str, fast_executemany=True, isolation_level='AUTOCOMMIT')
    engine.connect()

    # Set up logging
    start = timer()
    log_file = 'log.txt'
    logging.basicConfig(filename=log_file, level=logging.DEBUG)
    handler = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(handler)

    # Print startup messages
    logging.info(f'\nApplication started ... ({time.ctime()})')
    logging.info(f'\nCurrent working directory: {os.getcwd()}')
    logging.info('\nSetting up workspace...')

    # Read in all Resource Files
    xref_query = "SELECT Supplier, Supplier_ref FROM Revenew.dbo.crossref"
    xref_list = pd.read_sql(xref_query, engine)

    comm_query = "SELECT Supplier, Commodity FROM Revenew.dbo.commodities"
    comm_list = pd.read_sql(comm_query, engine)

    scorecard = pd.read_sql('SELECT * FROM Revenew.dbo.scorecard', engine)

    # Merge crossref and commodities
    comm_df = pd.merge(xref_list, comm_list, on=['Supplier'], how='left').groupby(
        ['Supplier_ref', 'Commodity']).size().reset_index(name='Freq')[['Supplier_ref', 'Commodity']]

    # clean up
    comm_df['Commodity'].replace(to_replace=['FACILITIES MAINTENANCE/SECURITY', 'REMOVE', 'STAFF AUGMENTATION',
                                             'INSPECTION/MONITORING/LAB SERVICES', 'TELECOMMUNICATIONS',
                                             'METER READING SERVICES', 'CHEMICALS/ADDITIVES/INDUSTRIAL GAS', ],
                                 value='SMALL_COUNT_COMM_GROUPS', inplace=True)

    # Read in the new client data
    logging.info(f'\nLoading new client data...')
    # Case 1: SPR Client
    if database is not None:
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
            SELECT COUNT(*) as Count FROM df             
"""
        input_df = pd.read_sql(dataquery, engine)
        input_df['Client'] = clientname

    # Case 2: Non-SPR Client, Rolled Up
    elif filename is not None:
        input_df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)
        expected_columns = ['Supplier', 'Total_Invoice_Amount', 'Total_Invoice_Count', 'Year']
        inlist = [col in input_df.columns for col in expected_columns]
        if sum(inlist) != len(expected_columns):
            missinglist = [col for col in expected_columns if col not in input_df.columns]
            logging.info(f'The following columns were expected but not found: {missinglist}..')
            raise SystemExit()
        input_df['Client'] = clientname

    # Case 3: Non-SPR Client, Raw
    elif filename2 is not None:
        temp_df = pd.read_csv(filename2, encoding='ISO-8859-1', low_memory=False)
        expected_columns = ['Vendor Name', 'Invoice Date', 'Gross Invoice Amount']
        inlist = [col in temp_df.columns for col in expected_columns]
        if sum(inlist) != len(expected_columns):
            missinglist = [col for col in expected_columns if col not in temp_df.columns]
            logging.info(f'The following columns were expected but not found: {missinglist}..')
            raise SystemExit()
        temp_df = temp_df[expected_columns].rename(columns={'Vendor Name': 'Supplier', })
        temp_df['Year'] = pd.to_datetime(temp_df['Invoice Date']).dt.year
        sums = temp_df.groupby(['Supplier', 'Year'], as_index=False)['Gross Invoice Amount'].sum()
        counts = temp_df.groupby(['Supplier', 'Year'], as_index=False)['Invoice Date'].agg(np.size)
        input_df = pd.merge(sums, counts, on=['Supplier', 'Year']).rename(
            columns={'Gross Invoice Amount': 'Total_Invoice_Amount', 'Invoice Date': 'Total_Invoice_Count', })
        input_df['Client'] = clientname

    # Null case
    else:
        logging.info('Sorry, something went wrong loading the new client data...')
        raise SystemExit()

    # Get average invoice amount
    logging.info('\nPreparing data for analysis...')
    input_df = input_df.dropna().reset_index(drop=True)
    input_df['Total_Invoice_Count'] = input_df.Total_Invoice_Count.fillna(0).astype(int, errors='ignore')
    input_df['Total_Invoice_Amount'] = input_df.Total_Invoice_Amount.fillna(0).astype(float, errors='ignore')
    input_df['Year'] = input_df.Year.fillna(0).astype(int).astype(str)  # filtered out zeros above so next part is ok
    input_df['Avg_Invoice_Size'] = input_df['Total_Invoice_Amount'].div(input_df['Total_Invoice_Count'])

    # Clean up the supplier name string
    input_df['Cleaned'] = [helpers.clean_up_string(s) for s in input_df.Supplier.fillna('')]

    # Create unique list of suppliers
    logging.info('\nCreating unique list of suppliers...')
    suppliers = input_df[['Supplier', 'Cleaned']].dropna().drop_duplicates()
    # suppliers.head(5)
    # xref_list.head(5)

    # Merge input data with crossref
    logging.info('\nMatching supplier names against cross-reference file...')
    combined = pd.merge(suppliers, xref_list, on='Supplier', how='outer', indicator=True)
    unmatched = combined[combined['_merge'] == 'left_only'].drop(columns=['_merge', 'Supplier_ref'])
    matched = combined[combined['_merge'] == 'both'].drop(columns=['_merge', 'Cleaned'])
    count_total, count_matched, count_unmatched = len(matched) + len(unmatched), len(matched), len(unmatched)
    # combined.head(5)
    # unmatched.head(5)
    # matched.head(5)

    # Print info about the matching
    logging.info(f'\tTotal suppliers: {count_total}')
    logging.info(f'\tMatched suppliers: {count_matched}')
    logging.info(f'\tUnmatched suppliers: {count_unmatched}')
    logging.info(f'\nTrying to soft-match the unmatched suppliers...')
    unmatched_series = unmatched['Cleaned']
    reference_series = comm_df['Supplier_ref']

    # Create master dict mapping unmatched suppliers to lists of candidates and their scores
    candidates = {}
    # Find candidate matches with score above threshold for each unmatched supplier
    for i, s in enumerate(unmatched_series):
        # Create a temp dict with {supplier_ref: fuzz_ ratio} key-value pairs
        d = {r: fuzz.ratio(s, r) for r in reference_series}
        # Keep if fuzz ratio above threshold
        if max(d.values()) > threshold:
            # Get the highest scoring supplier
            k = helpers.keys_with_top_values(d)
            # Update master dict with softmatch
            candidates[s] = k
        # Output progress through list of unmatched suppliers
        progress = round(100 * ((i + 1) / count_unmatched), 1)
        # Only print every 250 iterations
        if i % 250 == 0:
            out = f'{i} complete of {count_unmatched} ({progress}%)'
            logging.info(out)

    count_softmatch = len(candidates)
    logging.info(f'\tFound potential soft-matches for {count_softmatch} suppliers')

    #  Todo: deal with cases where there is more than one softmatch--now just taking the first one...
    match_dict = {item[0]: item[1][0] for item in candidates.items()}
    best_matches = pd.DataFrame(match_dict).T \
        .merge(suppliers, left_index=True, right_on='Cleaned') \
        .rename(columns={0: 'Supplier_ref', 1: 'Softmatch_Score'}) \
        .merge(input_df[['Cleaned', 'Total_Invoice_Amount']], on='Cleaned') \
        .groupby(['Supplier', 'Cleaned', 'Supplier_ref', ]) \
        .agg({'Softmatch_Score': 'min', 'Total_Invoice_Amount': 'sum'}) \
        .sort_values('Total_Invoice_Amount', ascending=False) \
        .reset_index()
    # best_matches.head(5)

    # Combine soft matches with unmatched suppliers
    if len(best_matches) > 0:
        soft_matched = best_matches[['Supplier', 'Supplier_ref']]
        # Add best matches back to supplier list
        xref = pd.concat([matched, soft_matched], axis=0, sort=True, ignore_index=True)
        xref = xref.merge(comm_df, on='Supplier_ref')
        final_df = input_df.merge(xref, on='Supplier', how='left').drop(columns='Cleaned')
    else:
        xref = matched.merge(comm_df, on='Supplier_ref')
        final_df = input_df.merge(xref, on='Supplier', how='left').drop(columns='Cleaned')

    # Update commodities that are missing because they were unmatched
    final_df['Commodity'].fillna('NOT_AVAILABLE', inplace=True)

    # Fill in Supplier_ref with blanks where not available
    final_df['Supplier_ref'].fillna('', inplace=True)

    # Only include supplier-years with invoices
    final_df = final_df.loc[final_df.Total_Invoice_Count > 0, :]

    # Scorecard computations
    logging.info('\nCalculating supplier scores based on scorecard...')

    # STEP 8b: do a full outer join with the scorecard
    final_df['key'] = 1
    scorecard['key'] = 1
    sc_df = pd.merge(final_df, scorecard, on='key').drop(columns='key')

    # STEP 8c-1: get the Spend sub-dataframe
    sc_df_spend = sc_df.loc[sc_df['Factor'] == 'Spend'].reset_index()
    sc_df_spend = (sc_df_spend.loc[sc_df_spend['Total_Invoice_Amount'] < sc_df_spend['Max']])
    sc_df_spend = (sc_df_spend.loc[sc_df_spend['Total_Invoice_Amount'] > sc_df_spend['Min']])
    sc_df_spend = sc_df_spend.reset_index().drop(columns=['Min', 'Max'])

    # STEP 8c-2: #get the InvoiceSize sub-dataframe
    sc_df_invoicesize = sc_df.loc[sc_df['Factor'] == 'InvoiceSize'].reset_index()
    sc_df_invoicesize = (sc_df_invoicesize.loc[sc_df_invoicesize['Avg_Invoice_Size'] < sc_df_invoicesize['Max']])
    sc_df_invoicesize = (sc_df_invoicesize.loc[sc_df_invoicesize['Avg_Invoice_Size'] > sc_df_invoicesize['Min']])
    sc_df_invoicesize = sc_df_invoicesize.reset_index().drop(columns=['Min', 'Max'])

    # STEP 8c-3: #get the InvoiceCount sub-dataframe
    sc_df_invoicecount = sc_df.loc[sc_df['Factor'] == 'InvoiceCount'].reset_index()
    sc_df_invoicecount = sc_df_invoicecount.loc[sc_df_invoicecount['Total_Invoice_Count'] < sc_df_invoicecount['Max']]
    sc_df_invoicecount = sc_df_invoicecount.loc[sc_df_invoicecount['Total_Invoice_Count'] > sc_df_invoicecount['Min']]
    sc_df_invoicecount = sc_df_invoicecount.reset_index().drop(columns=['Min', 'Max'])

    # STEP 8c-4: get the Commodity sub-dataframe
    sc_df_commodity = sc_df.loc[sc_df['Factor'] == 'Commodity'].reset_index()
    sc_df_commodity = (sc_df_commodity.loc[sc_df_commodity['Commodity'] == sc_df_commodity['Tier']])
    sc_df_commodity = sc_df_commodity.reset_index().drop(columns=['Min', 'Max'])

    # STEP 8d: # append all the factor scores
    scores = pd.concat(
        [sc_df_spend, sc_df_invoicesize, sc_df_invoicecount, sc_df_commodity],
        axis=0, sort=False, ignore_index=True) \
        .drop(columns=['level_0', 'index']).sort_values(['Supplier', 'Factor', 'Year'])

    # We were asked to re-capitalize supplier names
    scores['Supplier_ref'] = scores.Supplier_ref.str.upper()

    # Dealing with the weird way the scorecard calculations were originally done... reshaping and merging the data
    score_df = scores[['Client', 'Supplier', 'Supplier_ref', 'Year', 'Factor', 'Tier', 'Points']] \
        .set_index(['Client', 'Supplier', 'Supplier_ref', 'Year', 'Factor'])
    temp_df = final_df.rename(columns=
                              {'Total_Invoice_Amount': 'Spend', 'Total_Invoice_Count': 'InvoiceCount',
                               'Avg_Invoice_Size': 'InvoiceSize'}) \
        .drop(columns='key') \
        .replace('NOT_AVAILABLE', '') \
        .set_index(['Client', 'Supplier', 'Supplier_ref', 'Year'], verify_integrity=True) \
        .stack() \
        .reset_index() \
        .rename(columns={'level_4': 'Factor', 0: 'Value'}) \
        .set_index(['Client', 'Supplier', 'Supplier_ref', 'Year', 'Factor'], verify_integrity=True)
    out_df = temp_df.merge(score_df, left_index=True, right_index=True)
    out_df

    # score at supplier-year-factor-tier level  Fixme: remove min/max cols
    factor_scores = scores.groupby(['Supplier_ref', 'Year', 'Factor']).sum().stack().unstack()

    # score at supplier-year level
    year_scores = scores.groupby(['Supplier_ref', 'Year']).sum().stack().unstack().unstack(level=1)

    logging.info(f'\nWriting output file to {outputdir}...')

    # Create a Pandas Excel writer using XlSXWriter as the engine.
    writer = pd.ExcelWriter(f'{outputdir}/{clientname}_CC_Audit_Scorecard.xlsx', engine='xlsxwriter')
    input_df.to_excel(writer, sheet_name='Raw_Data', index=False)
    matched.to_excel(writer, sheet_name='CrossRef_Matched_Suppliers', index=False)
    unmatched.to_excel(writer, sheet_name='CrossRef_unMatched_Suppliers', index=False)
    best_matches.to_excel(writer, sheet_name='SoftMatched_Suppliers', index=False)
    out_df.to_excel(writer, sheet_name='SupplierScoreCard', index=False)
    factor_scores.to_excel(writer, sheet_name='Component_Scores')
    year_scores.to_excel(writer, sheet_name='Year_Scores')
    writer.save()

    # Stop timer
    end = timer()
    elapsed = end - start
    logging.info('\nDONE!\n\nApplication finished in {:.2f} seconds ... ({})'.format(elapsed, time.ctime()))


if __name__ == '__main__':
    main()
