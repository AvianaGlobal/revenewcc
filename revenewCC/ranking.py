#!/usr/bin/env python
import sys
from gooey import Gooey, GooeyParser


@Gooey(
    program_name='\nRevenewML\nCC Supplier Ranking\n',
    default_size=(700, 700),
    image_dir='::gooey/default',
    language_dir='gooey/languages',
)
def main():
    # Configure GUI
    parser = GooeyParser()
    parser.add_argument('dsn',
                        metavar='ODBC Data Source Name (DSN)',
                        help='Please enter the DSN below',
                        action='store', )
    parser.add_argument('clientname', metavar='CC Client Name',
                        help='Please enter the client\'s name below',
                        action='store', )
    parser.add_argument('outputdir', metavar='Output Data Folder',
                        help='Please select a target directory',
                        action='store',
                        widget='DirChooser')
    grp = parser.add_mutually_exclusive_group(required=True,
                                              gooey_options={'show_border': True})
    grp.add_argument('--database',
                     metavar='SPR Client',
                     widget='TextField',
                     help='Please enter the client database name',
                     action='store', )
    grp.add_argument('--filename',
                     metavar='NonSPR Client - Rolled Up',
                     widget='FileChooser',
                     help='CSV file '
                          'Columns: "Client", "Supplier", "Year", "Total_Invoice_Amount", "Total_Invoice_Count"',
                     action='store', )
    grp.add_argument('--filename2',
                     metavar='NonSPR Client - Raw',
                     widget='FileChooser',
                     help='CSV file '
                          'Columns: "Vendor Name", "Invoice Date", "Gross Invoice Amount"',
                     action='store', )
    args = parser.parse_args()
    dsn = args.dsn
    clientname = args.clientname
    database = args.database
    filename = args.filename
    filename2 = args.filename2
    outputdir = args.outputdir
    threshold = 85

    # Import packages
    import os
    import re
    import time
    import logging
    import numpy as np
    import pandas as pd
    from timeit import default_timer as timer
    from tqdm import tqdm

    # Set up logging
    start = timer()
    log_file = 'log.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO)
    handler = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(handler)
    logging.info(f'\nApplication started ... ({time.ctime()})')
    logging.info(f'\nCurrent working directory: {os.getcwd()}')

    # Step 0: Set up workspace
    logging.info('\nSetting up workspace...')

    def lower_case(str_to_lc):
        if type(str_to_lc) == str:
            return str_to_lc.lower()
        else:
            return str_to_lc

    def strip(str_to_lc):
        if type(str_to_lc) == str:
            return str_to_lc.strip()
        else:
            return str_to_lc

    def fuzz_ratio(str1, str2):
        from fuzzywuzzy import fuzz
        return fuzz.ratio(str1, str2)

    def remove_stuff_within_paranthesis(str1):
        if type(str1) == str:
            cleaned = (re.sub(r" ?\([^)]+\)", "", str1))
            return cleaned
        else:
            return str1

    def remove_substring_after_separator(str_to_clean, separator):
        if type(str_to_clean) == str:
            cleaned = str_to_clean.split(separator, 1)[0]
            return cleaned
        else:
            return str_to_clean

    def group_by_stats_list_sum(df, group_by_cols, val_cols):
        overall_grouped_df = pd.DataFrame()
        for i in (range(len(val_cols))):
            grouped_df = (df.groupby(group_by_cols)
                          .agg({val_cols[i]: [np.sum]})
                          .rename(columns={'sum': val_cols[i] + '_Sum'}))
            ret_df = grouped_df[val_cols[i]].copy()
            if i < 1:
                overall_grouped_df = ret_df.reset_index()
            else:
                overall_grouped_df = pd.merge(overall_grouped_df, ret_df.reset_index(), on=group_by_cols)
        return overall_grouped_df

    def sel_distinct(df, group_by_cols, sort_col, desc):
        orig_group_by_cols = group_by_cols[:]
        if desc == 1:
            group_by_cols.extend(sort_col)
            sorted_df = df.sort_values(by=group_by_cols, ascending=False)
        else:
            group_by_cols.extend(sort_col)
            sorted_df = df.sort_values(by=group_by_cols)
        sorted_df_first_row = sorted_df.groupby(orig_group_by_cols).first().reset_index()
        return sorted_df_first_row

    def group_by_stats_list_max(df, group_by_cols, val_cols):
        overall_grouped_df = pd.DataFrame()
        for i in (range(len(val_cols))):
            grouped_df = (df.groupby(group_by_cols)
                          .agg({val_cols[i]: [np.max]})
                          .rename(columns={'amax': val_cols[i] + '_Max'}))
            ret_df = grouped_df[val_cols[i]]
            if i < 1:
                overall_grouped_df = ret_df.reset_index()
            else:
                overall_grouped_df = pd.merge(overall_grouped_df, ret_df.reset_index(), on=group_by_cols, how='left')
        return overall_grouped_df

    def group_by_stats_list_min(df, group_by_cols, val_cols):
        overall_grouped_df = pd.DataFrame()
        for i in (range(len(val_cols))):
            grouped_df = (df.groupby(group_by_cols)
                          .agg({val_cols[i]: [np.min]})
                          .rename(columns={'amin': val_cols[i] + '_Min'}))
            ret_df = grouped_df[val_cols[i]]
            if i < 1:
                overall_grouped_df = ret_df.reset_index()
            else:
                overall_grouped_df = pd.merge(overall_grouped_df, ret_df.reset_index(), on=group_by_cols, how='left')
        return overall_grouped_df

    def clean_up_string(inp_string):
        cleaned = lower_case(inp_string)
        cleaned = remove_substring_after_separator(cleaned, "dba")
        cleaned = remove_substring_after_separator(cleaned, "-")
        cleaned = remove_stuff_within_paranthesis(cleaned)
        cleaned = strip(cleaned)
        cleaned = cleaned.replace('.', '')
        cleaned = cleaned.replace(',', '')
        cleaned = cleaned.replace('  ', ' ')
        cleaned = cleaned.replace('&', 'and')
        cleaned = cleaned.replace('-', '')
        cleaned = cleaned.replace(')', '')
        cleaned = cleaned.replace('*', '')
        cleaned = cleaned.replace('@', '')
        cleaned = cleaned.replace('#', '')
        cleaned = cleaned.replace('company', 'co')
        cleaned = cleaned.replace('corporation', 'corp')
        cleaned = cleaned.replace('incorporated', 'inc')
        cleaned = cleaned.replace('limited', 'ltd')
        # cleaned = cleaned.replace('co', '')
        # cleaned = cleaned.replace('corp', '')
        # cleaned = cleaned.replace('inc', '')
        # cleaned = cleaned.replace('ltd', '')
        # cleaned = cleaned.replace('pc', '')
        # cleaned = cleaned.replace('llc', '')
        # cleaned = cleaned.replace('llp ', '')
        cleaned = cleaned.replace('"', '')
        return cleaned

    # Work around macOS issues with ODBC
    if sys.platform == 'darwin':
        host = '208.43.250.18'
        port = '51949'
        user = 'sa'
        password = 'Aviana$92821'
        database = 'AvianaML'
        driver = '/usr/local/lib/libmsodbcsql.13.dylib'
        cnxn_str = f'mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver={driver}'
    else:
        cnxn_str = f'mssql+pyodbc://@{dsn}'  # Connection string assumes Windows auth

    # Make database connection engine
    from sqlalchemy import create_engine
    engine = create_engine(  # Options below are useful for debugging
        cnxn_str,
        fast_executemany=True,
        echo=False,
        # implicit_returning=False,
        # isolation_level="AUTOCOMMIT",
    )
    engine.connect()

    # Read in all Cross Reference Files
    logging.info('\nLoading cross-reference tables...')
    tqdm.pandas()
    supplier_crossref_list = pd.read_sql('SELECT Supplier, Supplier_ref FROM Revenew.dbo.crossref', engine, chunksize=1)
    supplier_crossref_list.progress_apply(lambda x: x, axis=1)
    # [x.progress_apply for x in supplier_crossref_list]

    # Use this in case you need to rewrite the table
    # supplier_crossref_list = pd.to_sql('crossref', engine, index=False, if_exists='replace', schema='Revenew.dbo')

    commodity_list = pd.read_sql('SELECT Supplier, Commodity FROM Revenew.dbo.commodities', engine, chunksize=10)
    commodity_list.progress_apply(lambda x: x, axis=1)
    # [x.progress_apply for x in commodity_list]
    # # Use this in case you need to rewrite the table
    # commodity_list.to_sql('commodities', engine, index=False, if_exists='replace', schema='Revenew.dbo')

    commodity_df = (pd.merge(supplier_crossref_list, commodity_list, on=['Supplier'], how='left')
                    .groupby(['Supplier_ref', 'Commodity']).size().reset_index(name='Freq')
                    )[['Supplier_ref', 'Commodity']]

    # Read in the new client data
    logging.info(f'\nLoading new client data...')

    # Case 1: SPR Client
    if database is not None:
        query = f"""
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
        input_df = pd.read_sql(query, engine)
        input_df['Client'] = clientname

    # Case 2: Non-SPR Client, Rolled Up
    elif filename is not None:
        input_df = pd.read_csv(filename, encoding='ISO-8859-1')
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
        temp_df['Client'] = clientname
        temp_df['Year'] = pd.to_datetime(temp_df['Invoice Date']).dt.year
        sums = temp_df.groupby(['Supplier', 'Year'], as_index=False)['Gross Invoice Amount'].sum()
        counts = temp_df.groupby(['Supplier', 'Year'], as_index=False)['Invoice Date'].agg(np.size)
        input_df = pd \
            .merge(sums, counts, on=['Supplier', 'Year']) \
            .rename(columns={'Gross Invoice Amount': 'Total_Invoice_Amount',
                             'Invoice Date': 'Total_Invoice_Count', })

    # Validate input
    else:
        input_df = None
    try:
        input_df
    except Exception:
        logging.info('Something went wrong loading the new client data...')
        raise SystemExit()

    ####################################
    # Data Processing Pipeline
    logging.info('\nPreparing data for analysis...')
    total = input_df['Total_Invoice_Amount']
    count = input_df['Total_Invoice_Count']
    input_df['Avg_Invoice_Size'] = total / count

    logging.info('\nCreating unique list of suppliers...')
    suppliers = pd.DataFrame({'Supplier': input_df['Supplier'].unique()})

    logging.info('\nMatching supplier names against cross-reference file...')
    input_df_with_ref = pd.merge(input_df, supplier_crossref_list, on='Supplier', how='left')

    logging.info('\nIdentifying non-matched suppliers...')
    unmatched = pd.DataFrame({
        'Supplier': input_df_with_ref[input_df_with_ref['Supplier_ref'].isnull()]['Supplier'].unique()})
    unmatched = unmatched.rename(columns={'Supplier': 'Unmatched_Supplier'}).reset_index()
    unmatched = unmatched.groupby(['Unmatched_Supplier']).size().reset_index(name='Freq')

    count_total = len(suppliers)
    count_matched = len(input_df_with_ref[input_df_with_ref['Supplier_ref'].notnull()]['Supplier_ref'].unique())
    count_unmatched = len(unmatched)

    logging.info(f'\nTotal suppliers: {count_total}')
    logging.info(f'Matched suppliers: {count_matched}')
    logging.info(f'Unmatched suppliers: {count_unmatched}')
    logging.info(f'\nTrying to soft-match the unmatched suppliers...')

    supplier_lookup = pd.DataFrame({'Supplier_ref': supplier_crossref_list['Supplier_ref'].unique()})
    unmatched_cross_ref = pd.merge(unmatched, supplier_lookup, on='key').drop('key', axis=1)

    # Create a cleaned version of Unmatched_Supplier
    unmatched_cross_ref['Unmatched_Supplier_Cleaned'] = [
        clean_up_string(s) for s in tqdm(unmatched_cross_ref.Unmatched_Supplier)]

    # Do the full softmatch, this will associate a MatchRatio to each possible match on the cleaned up version
    logging.info('\nEvaluating soft-matching scores...')
    name1 = 'Unmatched_Supplier_Cleaned'
    name2 = 'Supplier_ref'
    series1 = unmatched_cross_ref[name1]
    series2 = unmatched_cross_ref[name2]
    unmatched_cross_ref['MatchRatio'] = [fuzz_ratio(s1, s2) for s1, s2 in tqdm(zip(series1, series2))]

    ####################################
    # STEP 5c: Find the closest softmatch
    best_matches = sel_distinct(unmatched_cross_ref, ['Unmatched_Supplier'], ['MatchRatio'], 1)[
        ['Unmatched_Supplier', 'Supplier_ref', 'MatchRatio']]

    ####################################
    # STEP 5d: If > 85 match, then called matched
    soft_matches = best_matches.loc[best_matches['MatchRatio'] > threshold]
    soft_matches = (
        soft_matches[['Unmatched_Supplier', 'Supplier_ref', 'MatchRatio']].rename(
            columns={'Unmatched_Supplier': 'Supplier'}))
    # update the input_df_with_ref with these new softmatches
    input_df_with_ref.update(soft_matches)

    ####################################
    # STEP 5e: If <= 85 match, then called no soft matched
    no_soft_matches = best_matches.loc[best_matches['MatchRatio'] <= threshold][['Unmatched_Supplier']]

    ####################################
    # STEP 5f: consolidate the no_soft_matches (as they might have duplicates)
    no_soft_matches_1 = no_soft_matches.copy()
    no_soft_matches_2 = no_soft_matches.copy()

    no_soft_matches_2['key'] = 1
    no_soft_matches_1['key'] = 1
    no_soft_matches_2 = no_soft_matches_2.rename(columns={'Unmatched_Supplier': 'Unmatched_Supplier' + "_2"})

    no_soft_matches_full_join = pd.merge(no_soft_matches_1, no_soft_matches_2, on='key').drop('key', axis=1)
    no_soft_matches_full_join['MatchRatio'] = (
        np.vectorize(fuzz_ratio, otypes=[int])(no_soft_matches_full_join['Unmatched_Supplier'],
                                               no_soft_matches_full_join['Unmatched_Supplier' + "_2"]))  # fixme

    # pick close matches
    no_soft_matches_closematches = no_soft_matches_full_join.loc[(no_soft_matches_full_join['MatchRatio'] > 85)].copy()
    no_soft_matches_closematches = no_soft_matches_closematches.reset_index()
    no_soft_matches_closematches = no_soft_matches_closematches.sort_values(
        by=['Unmatched_Supplier', 'Unmatched_Supplier_2'])

    # for all the variants, pick the match with the longest (changed to shortest) variant
    no_soft_matches_cross_ref = no_soft_matches_closematches[
        ['Unmatched_Supplier', 'Unmatched_Supplier_2']].copy().reset_index()
    no_soft_matchesd_cross_ref = no_soft_matches_cross_ref.sort_values(
        by=['Unmatched_Supplier', 'Unmatched_Supplier_2'])
    no_soft_matches_cross_ref = group_by_stats_list_min(no_soft_matches_cross_ref, ['Unmatched_Supplier'],
                                                        ['Unmatched_Supplier_2'])
    no_soft_matches_cross_ref = no_soft_matches_cross_ref.rename(
        columns={'Unmatched_Supplier_2': 'Unmatched_Supplier_croosref'})

    no_soft_matches_cross_ref = no_soft_matches_cross_ref.reset_index()

    ####################################
    # STEP 8: bring in the commodity

    logging.info('\nAdding commodity type to supplier invoice data...')
    input_df_with_ref = (pd.merge(input_df_with_ref,
                                  commodity_df, on=['Supplier_ref'], how='left'))  # TODO investigate this join
    # fill_in when not available
    input_df_with_ref["Commodity"].fillna("NOT_AVAILABLE", inplace=True)

    # keep record of the original commodity before consolidating low count commodities for scoring
    input_df_with_ref["Original_Commodity"] = input_df_with_ref["Commodity"].copy()

    # fill_in small value commodities with SMALL_COUNT_COMM_GROUPS
    input_df_with_ref["Commodity"] = input_df_with_ref["Commodity"].replace("FACILITIES MAINTENANCE/SECURITY",
                                                                            "SMALL_COUNT_COMM_GROUPS")
    input_df_with_ref["Commodity"] = input_df_with_ref["Commodity"].replace("FACILITIES MAINTENANCE/SECURITY",
                                                                            "SMALL_COUNT_COMM_GROUPS")
    input_df_with_ref["Commodity"] = input_df_with_ref["Commodity"].replace("REMOVE", "SMALL_COUNT_COMM_GROUPS")
    input_df_with_ref["Commodity"] = input_df_with_ref["Commodity"].replace("STAFF AUGMENTATION",
                                                                            "SMALL_COUNT_COMM_GROUPS")
    input_df_with_ref["Commodity"] = input_df_with_ref["Commodity"].replace("INSPECTION/MONITORING/LAB SERVICES",
                                                                            "SMALL_COUNT_COMM_GROUPS")
    input_df_with_ref["Commodity"] = input_df_with_ref["Commodity"].replace("TELECOMMUNICATIONS",
                                                                            "SMALL_COUNT_COMM_GROUPS")
    input_df_with_ref["Commodity"] = input_df_with_ref["Commodity"].replace("METER READING SERVICES",
                                                                            "SMALL_COUNT_COMM_GROUPS")
    input_df_with_ref["Commodity"] = input_df_with_ref["Commodity"].replace("CHEMICALS/ADDITIVES/INDUSTRIAL GAS",
                                                                            "SMALL_COUNT_COMM_GROUPS")
    # input_df_with_ref.head(5)

    ###################################
    # Scorecard computations

    # STEP 8a: read in the scorecard
    logging.info('\nCalculating supplier scores based on scorecard...')
    supplier_scorecard = pd.read_sql('SELECT * FROM Revenew.dbo.scorecard', engine)
    # supplier_scorecard.to_sql('scorecard', con=engine, index=False, if_exists='replace', schema='Revenew.dbo')
    # supplier_scorecard.head(5)

    # STEP 8b: do a full outer join with the scorecard
    input_df_with_ref['key'] = 1
    supplier_scorecard['key'] = 1
    input_df_with_ref_with_scorecard = pd.merge(input_df_with_ref, supplier_scorecard, on='key').drop('key', axis=1)

    # STEP 8c-1: get the Spend sub-dataframe
    input_df_with_ref_with_scorecard_spend = (
        input_df_with_ref_with_scorecard.loc[input_df_with_ref_with_scorecard['Factor'] == 'Spend']).reset_index()

    # find the matching record
    input_df_with_ref_with_scorecard_spend = (
        input_df_with_ref_with_scorecard_spend.loc[input_df_with_ref_with_scorecard_spend['Total_Invoice_Amount']
                                                   < input_df_with_ref_with_scorecard_spend['Max']])

    input_df_with_ref_with_scorecard_spend = (
        input_df_with_ref_with_scorecard_spend.loc[input_df_with_ref_with_scorecard_spend['Total_Invoice_Amount']
                                                   > input_df_with_ref_with_scorecard_spend['Min']])

    input_df_with_ref_with_scorecard_spend = input_df_with_ref_with_scorecard_spend.reset_index()

    # STEP 8c-2: #get the InvoiceSize sub-dataframe
    input_df_with_ref_with_scorecard_invoicesize = (
        input_df_with_ref_with_scorecard.loc[input_df_with_ref_with_scorecard['Factor'] == 'InvoiceSize']).reset_index()

    # find the matching record
    input_df_with_ref_with_scorecard_invoicesize = (
        input_df_with_ref_with_scorecard_invoicesize.loc[
            input_df_with_ref_with_scorecard_invoicesize['Avg_Invoice_Size']
            < input_df_with_ref_with_scorecard_invoicesize['Max']])

    input_df_with_ref_with_scorecard_invoicesize = (
        input_df_with_ref_with_scorecard_invoicesize.loc[
            input_df_with_ref_with_scorecard_invoicesize['Avg_Invoice_Size']
            > input_df_with_ref_with_scorecard_invoicesize['Min']])

    input_df_with_ref_with_scorecard_invoicesize = input_df_with_ref_with_scorecard_invoicesize.reset_index()

    # STEP 8c-3: #get the InvoiceCount sub-dataframe
    input_df_with_ref_with_scorecard_invoicecount = (
        input_df_with_ref_with_scorecard.loc[
            input_df_with_ref_with_scorecard['Factor'] == 'InvoiceCount']).reset_index()

    # find the matching record
    input_df_with_ref_with_scorecard_invoicecount = (
        input_df_with_ref_with_scorecard_invoicecount.loc[
            input_df_with_ref_with_scorecard_invoicecount['Total_Invoice_Count']
            < input_df_with_ref_with_scorecard_invoicecount['Max']])

    input_df_with_ref_with_scorecard_invoicecount = (
        input_df_with_ref_with_scorecard_invoicecount.loc[
            input_df_with_ref_with_scorecard_invoicecount['Total_Invoice_Count']
            > input_df_with_ref_with_scorecard_invoicecount['Min']])

    input_df_with_ref_with_scorecard_invoicecount = input_df_with_ref_with_scorecard_invoicecount.reset_index()

    # STEP 8c-4: get the Commodity sub-dataframe
    input_df_with_ref_with_scorecard_commodity = (
        input_df_with_ref_with_scorecard.loc[input_df_with_ref_with_scorecard['Factor'] == 'Commodity']).reset_index()

    # find the matching record
    input_df_with_ref_with_scorecard_commodity = (
        input_df_with_ref_with_scorecard_commodity.loc[input_df_with_ref_with_scorecard_commodity['Commodity']
                                                       == input_df_with_ref_with_scorecard_commodity['Tier']])

    input_df_with_ref_with_scorecard_commodity = input_df_with_ref_with_scorecard_commodity.reset_index()

    # STEP 8d: # append all the factor scores
    scores = input_df_with_ref_with_scorecard_spend.append(input_df_with_ref_with_scorecard_invoicesize,
                                                           ignore_index=True)
    # scores=  scores.reset_index()
    scores = scores.append(input_df_with_ref_with_scorecard_invoicecount, ignore_index=True)
    # scores=  scores.reset_index()
    scores = scores.append(input_df_with_ref_with_scorecard_commodity, ignore_index=True)
    # scores=  scores.reset_index()

    # score at supplier-year-factor-tier level
    component_scores = scores.copy()
    # component_scores = scores[['Client', 'Supplier_ref', 'Year', 'Factor', 'Tier', 'Points']].copy()

    # score at supplier-year level
    scores = (group_by_stats_list_sum(scores, ['Client', 'Supplier_ref', 'Year'],
                                      ['Points'])
    [['Client', 'Supplier_ref', 'Year', 'Points_Sum']])

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

        Yr_data = (group_by_stats_list_max(Yr_data, ['Client', 'Supplier_ref', 'Year'],
                                           ['Total_Invoice_Amount', 'Total_Invoice_Count', 'Avg_Invoice_Size'])
        [['Client', 'Supplier_ref', 'Year', 'Total_Invoice_Amount_Max',
          'Total_Invoice_Count_Max', 'Avg_Invoice_Size_Max']])

        Yr_data = (Yr_data.rename
                   (columns={'Total_Invoice_Amount_Max': 'Total_Invoice_Amount',
                             'Total_Invoice_Count_Max': 'Total_Invoice_Count',
                             'Avg_Invoice_Size_Max': 'Avg_Invoice_Size'}))

        Yr_data_comm = (component_scores.loc[(component_scores['Year'] == Yr)
                                             & (component_scores['Factor'] == "Commodity")]
        [['Client', 'Supplier_ref', 'Year', 'Original_Commodity']])

        Yr_data = pd.merge(Yr_data, Yr_data_comm, on=['Client', 'Supplier_ref', 'Year'])

        Yr_score_data = scores.loc[scores['Year'] == Yr]
        Yr_data = pd.merge(Yr_data, Yr_score_data, on=['Client', 'Supplier_ref', 'Year'])
        Yr_data = Yr_data[
            ['Client', 'Supplier_ref', 'Original_Commodity', 'Total_Invoice_Amount', 'Total_Invoice_Count',
             'Avg_Invoice_Size', 'Points_Sum']]
        Yr_data = Yr_data.rename(
            columns={'Client': 'Client',
                     'Total_Invoice_Amount': str(int(Yr)) + "_Total_Invoice_Amount",
                     'Total_Invoice_Count': str(int(Yr)) + "_Total_Invoice_Count",
                     'Avg_Invoice_Size': str(int(Yr)) + "_Avg_Invoice_Size",
                     'Points_Sum': str(int(Yr)) + "_Score"
                     })

        Yr_data_List.append(Yr_data)
        i = i + 1

    final_data = Yr_data_List[0]
    i = 1
    while i < len(Yr_data_List):
        yr_df = Yr_data_List[i]
        yr_df = yr_df.drop(['Original_Commodity'], axis=1)

        final_data = (pd.merge(yr_df,
                               final_data, on=['Client', 'Supplier_ref']))
        i = i + 1

    # rearrange
    cols = list(final_data.columns.values)  # Make a list of all of the columns in the df
    cols.pop(cols.index('Client'))
    cols.pop(cols.index('Supplier_ref'))
    cols.pop(cols.index('Original_Commodity'))
    final_data = final_data[['Client', 'Supplier_ref',
                             'Original_Commodity'] + cols]  # Create new dataframe with columns in the order you want

    ####################################
    # STEP 10:  Write out all results into a spreadsheet
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(
        f'{outputdir}/CC_Audit_Scorecard.xlsx',
        engine='xlsxwriter')

    input_df.to_excel(writer, sheet_name='Raw_Data', index=False)
    matched.to_excel(writer, sheet_name='CrossRef_Matched_Suppliers', index=False)
    unmatched.to_excel(writer, sheet_name='CrossRef_unMatched_Suppliers', index=False)
    soft_matches.to_excel(writer, sheet_name='SoftMatched_Suppliers', index=False)
    no_soft_matches.to_excel(writer, sheet_name='NoSoft_Matched_Supp', index=False)
    no_soft_matches_cross_ref.to_excel(writer, sheet_name='NoSoft_Matched_Supp_Consol', index=False)
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
