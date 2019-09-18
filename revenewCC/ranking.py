#!/usr/bin/env python
from gooey import Gooey, GooeyParser


@Gooey(
    advanced=True,
    language='english',
    auto_start=False,
    target=None,
    program_name='\nRevenewML\nCC Supplier Ranking\n',
    program_description=None,
    default_size=(760, 620),
    use_legacy_titles=True,
    required_cols=1,
    optional_cols=1,
    dump_build_config=False,
    load_build_config=None,
    monospace_display=True,
    image_dir='::gooey/default',
    language_dir='./gooey/languages',
    progress_regex=None,
    progress_expr=None,
    disable_progress_bar_animation=False,
    disable_stop_button=False,
    group_by_type=False,
    header_height=80,
    navigation='SIDEBAR',
    tabbed_groups=False,
)
def main():
    # Configure GUI
    parser = GooeyParser()
    parser.add_argument('dsn', metavar='ODBC Data Source Name (DSN)',
                        help='Please enter the DSN below', action='store')
    parser.add_argument('outputdir', metavar='Output Data Folder',
                        help='Please select a target directory', action='store', widget='DirChooser')
    grp = parser.add_mutually_exclusive_group(required=True, gooey_options={'show_border': True})
    grp.add_argument('--database', metavar='SPR Client', widget='TextField',
                     help='Please enter the client database name', action='store')
    grp.add_argument('--filename', metavar='NonSPR Client - Rolled Up', widget='FileChooser',
                     help='CSV file '
                          '[Columns: Supplier, Year, Total_Invoice_Amount, Total_Invoice_Count]', action='store')
    grp.add_argument('--filename2', metavar='NonSPR Client - Not Rolled Up', widget='FileChooser',
                     help='CSV file '
                          '[Columns: Supplier, Invoice_Date, Gross_Invoice_Amount]', action='store')
    args = parser.parse_args()
    dsn = args.dsn
    database = args.database
    filename = args.filename
    filename2 = args.filename2
    outputdir = args.outputdir

    # Import packages
    import os
    import time
    import logging
    from timeit import default_timer as timer

    # Get application path
    application_path = os.getcwd()

    # Set up logging
    start = timer()
    log_file = application_path + '/../log.txt'
    logging.basicConfig(filename=log_file, level=logging.INFO)
    handler = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(handler)
    logging.info(f'\nApplication started ... ({time.ctime()})')
    logging.info(f'\nCurrent working directory: {application_path}')

    # Step 0: Set up workspace
    import pandas as pd
    import numpy as np
    import itertools
    import sqlite3
    import fuzzywuzzy
    import re

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

    def cartesian(df1, df2):
        rows = itertools.product(df1.iterrows(), df2.iterrows())
        df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
        return df.reset_index(drop=True)

    def fuzz_partial_ratio(str1, str2):
        return fuzzywuzzy.partial_ratio(str1, str2)

    def fuzz_token_set_ratio(str1, str2):
        return fuzzywuzzy.partial_ratio(str1, str2)

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

    def strip_numbers(str1):
        cleaned = re.sub('[0-9]+', '', str1)
        return cleaned

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

    def group_by_count(df, group_by_cols):
        grouped_df = df.groupby(group_by_cols).size().reset_index(name='Count')
        return grouped_df

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

    def get_string(str1, _from, _to):
        end_from = str1.find(_from) + len(_from)
        return str1[end_from: str1.find(_to, end_from)]

    def clean_up_string(inp_string):
        cleaned = inp_string
        # convert to lower case
        if type(inp_string) == str:
            cleaned = lower_case(inp_string)
        # strip anything after dba
        cleaned = remove_substring_after_separator(cleaned, "dba")
        # strip anything after "-"
        remove_substring_after_separator(cleaned, "-")
        # remove substrings within paranthesis
        cleaned = remove_stuff_within_paranthesis(cleaned)
        # clean up the lcase versions
        cleaned = strip(cleaned)
        # standardize
        cleaned = cleaned.replace('&', 'and')
        cleaned = cleaned.replace('co.', '')
        cleaned = cleaned.replace('.', '')
        cleaned = cleaned.replace(',', '')
        cleaned = cleaned.replace(' llc', '')
        cleaned = cleaned.replace(' llp ', '')
        cleaned = cleaned.replace('company', '')
        cleaned = cleaned.replace('corporation', '')
        cleaned = cleaned.replace(' inc', '')
        cleaned = cleaned.replace(' ltd', '')
        cleaned = cleaned.replace(' corp', '')
        cleaned = cleaned.replace('pc', '')
        cleaned = cleaned.replace('-', '')
        cleaned = cleaned.replace(')', '')
        cleaned = cleaned.replace('*', '')
        cleaned = cleaned.replace('@', '')
        cleaned = cleaned.replace('#', '')
        cleaned = strip(cleaned)
        return cleaned

    cnxn_str = f'mssql+pyodbc://@{dsn}'
 
    # Make database connection engine
    from sqlalchemy import create_engine
    engine = create_engine(
        cnxn_str,
        #fast_executemany=True,
        #echo=False,
        #implicit_returning=False,
        #isolation_level="AUTOCOMMIT",
    )
    engine.connect()

    # Read in all Cross Reference Files
    print('\nReading in supplier cross-reference files...')

    supplier_crossref_list = pd.read_sql('select Supplier, Supplier_ref from RevenewCC.dbo.crossref', engine)
    # supplier_crossref_list = pd.to_sql('crossref', engine, index=False, if_exists='replace', schema='RevenewCC.dbo')
    
    commodity_list = pd.read_sql('select Supplier, Commodity from RevenewCC.dbo.commodities', engine)
    # commodity_list.to_sql('commodities', engine, index=False, if_exists='replace', schema='RevenewCC.dbo')
    
    commodity_df = (
        pd.merge(
            supplier_crossref_list,
            commodity_list,
            on=['Supplier'], how='left'
        ).groupby(['Supplier_ref', 'Commodity']).size().reset_index(name='Freq')
    )[['Supplier_ref', 'Commodity']]
    
    print(commodity_df.sample(5))

    # Read in the new client data
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
        input_df['Client'] = database
    elif filename is not None:  # TODO Add data validation checks
        # Expects: (Supplier, Total_Invoice_Amount, Total_Invoice_Count, Year)
        input_df = pd.read_csv(filename, encoding='ISO-8859-1', sep='\t')
        input_df['Supplier'] = input_df['Supplier'].astype(str)
        input_df['Client'] = input_df['Client'].astype(str)
    elif filename2 is not None:  # TODO Add data validation checks
        # Expects: (Supplier, Total_Invoice_Amount, Total_Invoice_Count, Year)
        input_df = pd.read_csv(filename2, encoding='ISO-8859-1', sep='\t')
        input_df['Supplier'] = input_df['Supplier'].astype(str)
        input_df['Client'] = input_df['Client'].astype(str)

    # Data Processing Pipeline
    print('\nPreparing data for analysis...')

    ####################################
    # STEP 3:  Merged and find direct matches
    input_df_with_ref = (pd.merge(input_df, supplier_crossref_list, on='Supplier', how='left')
    [['Supplier', 'Total_Invoice_Amount', 'Total_Invoice_Count', 'Year', 'Client', 'Supplier_ref']])
    matched = input_df_with_ref[
        input_df_with_ref['Supplier_ref'].notnull()]  # has Supplier_ref added to the standard dataframe

    ####################################
    # STEP 4:  what didnt find a match with the cross reference file
    unmatched = input_df_with_ref[input_df_with_ref['Supplier_ref'].isnull()][['Supplier']]
    unmatched = unmatched.rename(columns={'Supplier': 'Unmatched_Supplier'}).reset_index()
    unmatched = unmatched.groupby(['Unmatched_Supplier']).size().reset_index(name='Freq')

    ####################################
    # Step 5: Try to softmatch the unmatched

    # STEP 5a: First step is to create a full cross match of unmatched and supplier crossref
    unmatched['key'] = 1
    supplier_crossref_list['key'] = 1
    unmatched_cross_ref = pd.merge(unmatched, supplier_crossref_list, on='key').drop('key', axis=1)

    # first create a "cleaned" version of Unmatched_Supplier
    unmatched_cross_ref['Unmatched_Supplier_Cleaned'] = unmatched_cross_ref['Unmatched_Supplier'].copy().astype(str)
    unmatched_cross_ref['Unmatched_Supplier_Cleaned'] = np.vectorize(clean_up_string, otypes=[str])(
        unmatched_cross_ref['Unmatched_Supplier_Cleaned'])

    # In[335]:

    ####################################
    # STEP 5b: Do the full softmatch, this will associate a MatchRatio to each possible match of Unmatched_Supplier_Cleaned & Supplier_Cleaned
    # doing on the cleaned up version
    name1 = 'Unmatched_Supplier_Cleaned'
    name2 = 'Supplier_ref'
    unmatched_cross_ref['Unmatched_Supplier_Cleaned'] = unmatched_cross_ref['Unmatched_Supplier_Cleaned'].astype(str)
    unmatched_cross_ref['Supplier_ref'] = unmatched_cross_ref['Supplier_ref'].astype(str)
    # get the match ratio
    unmatched_cross_ref['MatchRatio'] = np.vectorize(fuzz_ratio, otypes=[int])(unmatched_cross_ref[name1],
                                                                               unmatched_cross_ref[name2])

    # In[336]:

    ####################################
    # STEP 5c: Find the closest softmatch
    best_matches = sel_distinct(unmatched_cross_ref, ['Unmatched_Supplier'], ['MatchRatio'], 1)[
        ['Unmatched_Supplier', 'Supplier_ref', 'MatchRatio']]

    # In[337]:

    ####################################
    # STEP 5d: If > 85 match, then called matched
    soft_matches = best_matches.loc[best_matches['MatchRatio'] > 85]
    soft_matches = (
        soft_matches[['Unmatched_Supplier', 'Supplier_ref', 'MatchRatio']].rename(
            columns={'Unmatched_Supplier': 'Supplier'}))
    # update the input_df_with_ref with these new softmatches
    input_df_with_ref.update(soft_matches)

    # In[338]:

    ####################################
    # STEP 5e: If <= 85 match, then called no soft matched
    no_soft_matches = best_matches.loc[best_matches['MatchRatio'] <= 85][['Unmatched_Supplier']]

    # In[339]:

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
                                               no_soft_matches_full_join['Unmatched_Supplier' + "_2"]))

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

    # In[340]:

    ####################################
    # STEP 6: Aggregate to Client, Supplier, Year level. TODO Check this
    # Even if the raw data is at the invoice level, this makese sure that the data is rolled to supplier-year level
    input_df_with_ref = group_by_stats_list_sum(
        input_df_with_ref,
        ['Client', 'Supplier_ref', 'Year'],
        ['Total_Invoice_Amount', 'Total_Invoice_Count']
    )[['Client', 'Supplier_ref', 'Year', 'Total_Invoice_Amount_Sum', 'Total_Invoice_Count_Sum']]

    # In[341]:

    ####################################
    # STEP 7:  Create new column Average Invoice Size
    input_df_with_ref = input_df_with_ref.rename(columns={'Total_Invoice_Amount_Sum': 'Total_Invoice_Amount'})
    input_df_with_ref = input_df_with_ref.rename(columns={'Total_Invoice_Count_Sum': 'Total_Invoice_Count'})
    input_df_with_ref['Avg_Invoice_Size'] = input_df_with_ref['Total_Invoice_Amount'] / input_df_with_ref[
        'Total_Invoice_Count']

    # In[342]:
    input_df_with_ref.head(5)
    ####################################
    # STEP 8: bring in the commodity
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
    input_df_with_ref.head(5)

    ###################################
    # STEP 8: Scorecard computations

    ####################################
    # STEP 8a: read in the scorecard
    print('\nCalculating supplier scores based on scorecard...')
    supplier_scorecard = pd.read_sql('select * from RevenewCC.dbo.scorecard', engine)
    supplier_scorecard.head(5)

    # supplier_scorecard.to_sql('scorecard', con=engine, index=False, if_exists='replace', schema='RevenewCC.dbo')
    ####################################
    # STEP 8b: do a full outer join with the scorecard
    input_df_with_ref['key'] = 1
    supplier_scorecard['key'] = 1
    input_df_with_ref_with_scorecard = pd.merge(input_df_with_ref, supplier_scorecard, on='key').drop('key', axis=1)
  
    ####################################
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

    ####################################
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

    ####################################
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

    ####################################
    # STEP 8c-4: get the Commodity sub-dataframe
    input_df_with_ref_with_scorecard_commodity = (
        input_df_with_ref_with_scorecard.loc[input_df_with_ref_with_scorecard['Factor'] == 'Commodity']).reset_index()

    # find the matching record
    input_df_with_ref_with_scorecard_commodity = (
        input_df_with_ref_with_scorecard_commodity.loc[input_df_with_ref_with_scorecard_commodity['Commodity']
                                                       == input_df_with_ref_with_scorecard_commodity['Tier']])

    input_df_with_ref_with_scorecard_commodity = input_df_with_ref_with_scorecard_commodity.reset_index()

    ####################################
    # STEP 8d: # append all the factor scores
    scores = input_df_with_ref_with_scorecard_spend.append(input_df_with_ref_with_scorecard_invoicesize,
                                                           ignore_index=True)
    # scores=  scores.reset_index()
    scores = scores.append(input_df_with_ref_with_scorecard_invoicecount, ignore_index=True)
    # scores=  scores.reset_index()
    scores = scores.append(input_df_with_ref_with_scorecard_commodity, ignore_index=True)
    # scores=  scores.reset_index()
    scores.head(5)

    # score at supplier-year-factor-tier level
    component_scores = scores.copy()
    # component_scores = scores[['Client', 'Supplier_ref', 'Year', 'Factor', 'Tier', 'Points']].copy()

    # score at supplier-year level
    scores = (group_by_stats_list_sum(scores, ['Client', 'Supplier_ref', 'Year'],
                                      ['Points'])
    [['Client', 'Supplier_ref', 'Year', 'Points_Sum']])

    print(f'\nWriting output file to {outputdir}...')

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

    input_df.to_excel(writer, sheet_name='Raw_Data')
    matched.to_excel(writer, sheet_name='CrossRef_Matched_Suppliers')
    unmatched.to_excel(writer, sheet_name='CrossRef_unMatched_Suppliers')
    soft_matches.to_excel(writer, sheet_name='SoftMatched_Suppliers')
    no_soft_matches.to_excel(writer, sheet_name='NoSoft_Matched_Supp')
    no_soft_matches_cross_ref.to_excel(writer, sheet_name='NoSoft_Matched_Supp_Consol')
    component_scores.to_excel(writer, sheet_name='Component_Scores')
    scores.to_excel(writer, sheet_name='SupplierScoreCard')
    final_data.to_excel(writer, sheet_name='FinalScorecard')
    writer.save()

    # Stop timer
    end = timer()
    elapsed = end - start
    logging.info('\nDONE!\n\nApplication finished in {:.2f} seconds ... ({})'.format(elapsed, time.ctime()))


if __name__ == '__main__':
    main()
