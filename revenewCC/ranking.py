#!/usr/bin/env python
from gooey import Gooey


@Gooey(program_name='\nRevenewML\nCC Supplier Ranking\n', default_size=(700, 700), image_dir='::gooey/default',
       language_dir='gooey/languages', )
def main():
    from revenewCC.argparser import parser
    args = parser.parse_args()
    dsn = args.dsn
    clientname = args.clientname
    database = args.database
    filename = args.filename
    filename2 = args.filename2
    outputdir = args.outputdir

    threshold = 89

    import os
    import time
    import logging
    import numpy as np
    import pandas as pd
    from fuzzywuzzy import fuzz
    from timeit import default_timer as timer

    # ### For progress bar ###
    #
    # import io
    # from contextlib import redirect_stderr
    # from tqdm.auto import tqdm
    #

    from revenewCC import helpers
    from revenewCC.dbconnect import dbconnect
    from revenewCC.defaults import database, clientname, filename, filename2, outputdir

    # Set up data connection
    # engine = dbconnect()

    # Set up logging
    start = timer()
    log_file = 'log.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO)
    handler = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(handler)

    # Print startup messages
    logging.info(f'\nApplication started ... ({time.ctime()})')
    logging.info(f'\nCurrent working directory: {os.getcwd()}')
    logging.info('\nSetting up workspace...')

    # Read in all Resource Files
    supplier_crossref_list = pd.read_pickle('revenewCC/inputdata/crossref.pkl')
    commodity_list = pd.read_pickle('revenewCC/inputdata/commodities.pkl')
    supplier_scorecard = pd.read_pickle('revenewCC/inputdata/scorecard.pkl')

    # Merge crossref and commodities
    commodity_df = pd.merge(supplier_crossref_list, commodity_list, on=['Supplier'], how='left').groupby(
        ['Supplier_ref', 'Commodity']).size().reset_index(name='Freq')[['Supplier_ref', 'Commodity']]

    # fill_in when not available
    commodity_df["Commodity"].fillna("NOT_AVAILABLE", inplace=True)

    # fill_in small value commodities with SMALL_COUNT_COMM_GROUPS
    commodity_df["Commodity"].replace(to_replace=["FACILITIES MAINTENANCE/SECURITY", "REMOVE", "STAFF AUGMENTATION",
                                                  "INSPECTION/MONITORING/LAB SERVICES", "TELECOMMUNICATIONS",
                                                  "METER READING SERVICES", "CHEMICALS/ADDITIVES/INDUSTRIAL GAS", ],
                                      value="SMALL_COUNT_COMM_GROUPS", inplace=True)

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
        input_df = pd.DataFrame()
        chunks = pd.read_sql(dataquery, engine, chunksize=1)
        # count = pd.read_sql(countquery, engine).values[0][0]
        # f = io.StringIO()
        # with redirect_stderr(f):
        # for chunk in tqdm(chunks, total=count):
        for chunk in chunks:
            input_df = pd.concat(
                [input_df, chunk])  # prog = f.getvalue().split('\r ')[-1].strip()  # print(prog)  # time.sleep(0.2)
        input_df['Client'] = clientname
    # Case 2: Non-SPR Client, Rolled Up
    elif filename is not None:
        input_df = pd.DataFrame()
        chunks = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False, chunksize=1, na_values='')
        # f = io.StringIO()
        # with redirect_stderr(f):
        for chunk in chunks:
            input_df = pd.concat(
                [input_df, chunk])  # prog = f.getvalue().split('\r ')[-1].strip()  # print(prog)  # time.sleep(0.2)
        expected_columns = ['Supplier', 'Total_Invoice_Amount', 'Total_Invoice_Count', 'Year']
        inlist = [col in input_df.columns for col in expected_columns]
        if sum(inlist) != len(expected_columns):
            missinglist = [col for col in expected_columns if col not in input_df.columns]
            logging.info(f'The following columns were expected but not found: {missinglist}..')
            raise SystemExit()
        input_df['Client'] = clientname
    # Case 3: Non-SPR Client, Raw
    elif filename2 is not None:
        temp_df = pd.DataFrame()
        chunks = pd.read_csv(filename2, encoding='ISO-8859-1', low_memory=False, chunksize=1)
        # f = io.StringIO()
        # with redirect_stderr(f):
        for chunk in chunks:
            temp_df = pd.concat(
                [temp_df, chunk])  # prog = f.getvalue().split('\r ')[-1].strip()  # print(prog)  # time.sleep(0.2)
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

    # Data processing
    logging.info('\nPreparing data for analysis...')
    input_df = input_df.dropna().reset_index(drop=True)
    input_df['Total_Invoice_Count'] = input_df.Total_Invoice_Count.fillna(0).astype(int)
    input_df['Year'] = input_df.Year.fillna(0).astype(int).astype(str)
    input_df['Avg_Invoice_Size'] = input_df['Total_Invoice_Amount'] / input_df['Total_Invoice_Count']

    # Clean up the supplier name string
    input_df['Cleaned'] = [helpers.clean_up_string(s) for s in input_df.Supplier.fillna('')]

    # Create unique list of suppliers
    logging.info('\nCreating unique list of suppliers...')
    suppliers = input_df[['Supplier', 'Cleaned']].dropna().drop_duplicates()

    # Merge input data with crossref
    logging.info('\nMatching supplier names against cross-reference file...')

    combined = pd.merge(suppliers, supplier_crossref_list, on='Supplier', how='outer', indicator=True)
    unmatched = combined[combined['_merge'] == 'left_only'].drop(columns=['_merge', 'Supplier_ref'])
    matched = combined[combined['_merge'] == 'both'].drop(columns=['_merge', 'Cleaned'])
    count_total, count_matched, count_unmatched = len(suppliers), len(matched), len(unmatched)

    # Print info about the matching
    logging.info(f'\tTotal suppliers: {count_total}')
    logging.info(f'\tMatched suppliers: {count_matched}')
    logging.info(f'\tUnmatched suppliers: {count_unmatched}')
    logging.info(f'\nTrying to soft-match the unmatched suppliers...')

    unmatched_series = unmatched['Cleaned']
    reference_series = commodity_df['Supplier_ref']

    # Find candidate matches with score above threshold
    candidates = {}
    for i, s in enumerate(unmatched_series):
        prog = round(100 * ((i + 1) / count_unmatched), 2)
        d = {r: fuzz.ratio(s, r) for r in reference_series}
        if max(d.values()) > threshold:
            k = helpers.keys_with_top_values(d)
            # print(f'{s} = {k[0][0]}...?')
            candidates[s] = k  # print(f'{i + 1}/{count_unmatched} ({prog}%)', end='\r', flush=True)

    count_softmatch = len(candidates)
    logging.info(f'\tFound potential soft-matches for {count_softmatch} supplier(s)')

    #  TODO deal with cases where there is more than one softmatch--just taking the first one for now
    best_matches = pd.DataFrame({item[0]: item[1][0] for item in candidates.items()}).T.merge(suppliers,
                                                                                              left_index=True,
                                                                                              right_on='Cleaned').rename(
        columns={0: 'Supplier_ref', 1: 'Softmatch_Score'})

    # Combine softmatches with unmatched suppliers
    soft_matched = unmatched.merge(best_matches[['Supplier', 'Supplier_ref']], on='Supplier', how='left')
    # soft_matched['Supplier_ref'].fillna(value=soft_matched['Cleaned'], inplace=True)
    soft_matched.drop(columns='Cleaned', inplace=True)

    # Add best matches back to supplier list
    xref = pd.concat([matched, soft_matched], axis=0, sort=True, ignore_index=True).merge(commodity_df,
                                                                                          on='Supplier_ref')

    len(xref.Supplier.unique())
    len(xref.Supplier_ref.unique())

    final_df = input_df.merge(xref, on='Supplier').drop(columns='Cleaned')[
        ['Client', 'Supplier', 'Supplier_ref', 'Commodity', 'Year', 'Total_Invoice_Amount', 'Total_Invoice_Count',
         'Avg_Invoice_Size', ]]

    # Scorecard computations
    logging.info('\nCalculating supplier scores based on scorecard...')

    # STEP 8b: do a full outer join with the scorecard
    final_df['key'] = 1
    supplier_scorecard['key'] = 1
    scorecard_df = pd.merge(final_df, supplier_scorecard, on='key').drop('key', axis=1)

    # STEP 8c-1: get the Spend sub-dataframe
    scorecard_df_spend = (scorecard_df.loc[scorecard_df['Factor'] == 'Spend']).reset_index()

    # find the matching record
    scorecard_df_spend = (
        scorecard_df_spend.loc[scorecard_df_spend['Total_Invoice_Amount'] < scorecard_df_spend['Max']])

    scorecard_df_spend = (
        scorecard_df_spend.loc[scorecard_df_spend['Total_Invoice_Amount'] > scorecard_df_spend['Min']])

    scorecard_df_spend = scorecard_df_spend.reset_index()

    # STEP 8c-2: #get the InvoiceSize sub-dataframe
    scorecard_df_invoicesize = (scorecard_df.loc[scorecard_df['Factor'] == 'InvoiceSize']).reset_index()

    # find the matching record
    scorecard_df_invoicesize = (
        scorecard_df_invoicesize.loc[scorecard_df_invoicesize['Avg_Invoice_Size'] < scorecard_df_invoicesize['Max']])

    scorecard_df_invoicesize = (
        scorecard_df_invoicesize.loc[scorecard_df_invoicesize['Avg_Invoice_Size'] > scorecard_df_invoicesize['Min']])

    scorecard_df_invoicesize = scorecard_df_invoicesize.reset_index()

    # STEP 8c-3: #get the InvoiceCount sub-dataframe
    scorecard_df_invoicecount = (scorecard_df.loc[scorecard_df['Factor'] == 'InvoiceCount']).reset_index()

    # find the matching record
    scorecard_df_invoicecount = (scorecard_df_invoicecount.loc[
        scorecard_df_invoicecount['Total_Invoice_Count'] < scorecard_df_invoicecount['Max']])

    scorecard_df_invoicecount = (scorecard_df_invoicecount.loc[
        scorecard_df_invoicecount['Total_Invoice_Count'] > scorecard_df_invoicecount['Min']])

    scorecard_df_invoicecount = scorecard_df_invoicecount.reset_index()

    # STEP 8c-4: get the Commodity sub-dataframe
    scorecard_df_commodity = (scorecard_df.loc[scorecard_df['Factor'] == 'Commodity']).reset_index()

    # find the matching record
    scorecard_df_commodity = (
        scorecard_df_commodity.loc[scorecard_df_commodity['Commodity'] == scorecard_df_commodity['Tier']])

    scorecard_df_commodity = scorecard_df_commodity.reset_index()

    # STEP 8d: # append all the factor scores
    scores = scorecard_df_spend.append(scorecard_df_invoicesize, ignore_index=True)
    scores = scores.append(scorecard_df_invoicecount, ignore_index=True)
    scores = scores.append(scorecard_df_commodity, ignore_index=True)

    # score at supplier-year-factor-tier level
    component_scores = ...

    # score at supplier-year level
    scores = ...
    logging.info(f'\nWriting output file to {outputdir}...')

    ####################################
    # STEP 9:  format in the way scorecard is currently implemented
    num_years = len(component_scores['Year'].unique())
    years = component_scores['Year'].unique()
    years = np.sort(years)

    Yr_data_List = list()

    i = 0
    while i < num_years:
        Yr = years[i]
        Yr_data = component_scores.loc[component_scores['Year'] == Yr]

        Yr_data = (helpers.group_by_stats_list_max(Yr_data, ['Client', 'Supplier_ref', 'Year'],
                                                   ['Total_Invoice_Amount', 'Total_Invoice_Count', 'Avg_Invoice_Size'])[
            ['Client', 'Supplier_ref', 'Year', 'Total_Invoice_Amount_Max', 'Total_Invoice_Count_Max',
             'Avg_Invoice_Size_Max']])

        Yr_data = (Yr_data.rename(columns={'Total_Invoice_Amount_Max': 'Total_Invoice_Amount',
                                           'Total_Invoice_Count_Max': 'Total_Invoice_Count',
                                           'Avg_Invoice_Size_Max': 'Avg_Invoice_Size'}))

        Yr_data_comm = (
            component_scores.loc[(component_scores['Year'] == Yr) & (component_scores['Factor'] == "Commodity")][
                ['Client', 'Supplier_ref', 'Year', 'Original_Commodity']])

        Yr_data = pd.merge(Yr_data, Yr_data_comm, on=['Client', 'Supplier_ref', 'Year'])

        Yr_score_data = scores.loc[scores['Year'] == Yr]
        Yr_data = pd.merge(Yr_data, Yr_score_data, on=['Client', 'Supplier_ref', 'Year'])
        Yr_data = Yr_data[
            ['Client', 'Supplier_ref', 'Original_Commodity', 'Total_Invoice_Amount', 'Total_Invoice_Count',
             'Avg_Invoice_Size', 'Points_Sum']]
        Yr_data = Yr_data.rename(
            columns={'Client': 'Client', 'Total_Invoice_Amount': str(int(Yr)) + "_Total_Invoice_Amount",
                     'Total_Invoice_Count': str(int(Yr)) + "_Total_Invoice_Count",
                     'Avg_Invoice_Size': str(int(Yr)) + "_Avg_Invoice_Size", 'Points_Sum': str(int(Yr)) + "_Score"})

        Yr_data_List.append(Yr_data)
        i = i + 1

    final_data = Yr_data_List[0]
    i = 1
    while i < len(Yr_data_List):
        yr_df = Yr_data_List[i]
        yr_df = yr_df.drop(['Original_Commodity'], axis=1)

        final_data = (pd.merge(yr_df, final_data, on=['Client', 'Supplier_ref']))
        i = i + 1

    # Create new dataframe with columns in the order you want
    cols = list(final_data.columns.values)  # Make a list of all of the columns in the df
    cols.pop(cols.index('Client'))
    cols.pop(cols.index('Supplier_ref'))
    cols.pop(cols.index('Original_Commodity'))
    final_data = final_data[['Client', 'Supplier_ref', 'Original_Commodity'] + cols]

    ####################################
    # STEP 10:  Write out all results into a spreadsheet
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(f'{outputdir}/CC_Audit_Scorecard.xlsx', engine='xlsxwriter')

    input_df.to_excel(writer, sheet_name='Raw_Data', index=False)
    matched.to_excel(writer, sheet_name='CrossRef_Matched_Suppliers', index=False)
    unmatched.to_excel(writer, sheet_name='CrossRef_unMatched_Suppliers', index=False)
    best_matches.to_excel(writer, sheet_name='SoftMatched_Suppliers', index=False)
    unmatched.to_excel(writer, sheet_name='NoSoft_Matched_Supp', index=False)
    component_scores.to_excel(writer, sheet_name='Component_Scores', index=False)
    scores.to_excel(writer, sheet_name='SupplierScoreCard', index=False)
    final_data.to_excel(writer, sheet_name='FinalScorecard', index=False)
    writer.save()

    # Stop timer
    end = timer()
    elapsed = end - start
    logging.info('\nDONE!\n\nApplication finished in {:.2f} seconds ... ({})'.format(elapsed, time.ctime()))


if __name__ == '__main__':
    main()
