#!/usr/bin/env python
from gooey import Gooey, GooeyParser


@Gooey(program_name='\nRevenewML\nCC Supplier Ranking\n',
    #    dump_build_config=True,
       load_build_config='gooey_configw.json',
    #    default_size=(760, 540)
    )
def main():

    # Configure GUI
    parser = GooeyParser()
    parser.add_argument('dsn', metavar='ODBC Data Source Name (DSN)',
                        help='Please enter the DSN below', action='store')
    grp = parser.add_mutually_exclusive_group(required=True, gooey_options={'show_border': True})
    grp.add_argument('--database', metavar='SPR Client', widget='TextField',
                     help='Please enter the client database name', action='store')
    grp.add_argument('--filename', metavar='NonSPR Client', widget='FileChooser',
                     help='Please locate the client data file', action='store')
    args = parser.parse_args()
    dsn = args.dsn
    database = args.database
    filename = args.filename

    # Import packages
    import os
    import time
    import logging
    import numpy as np
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
    logging.info('\n<=============================================================>')
    logging.info(f'\nApplication path: {application_path}')
    logging.info(f'\nCurrent working directory: {os.getcwd()}')
    logging.info(f'\nApplication started ... ({time.ctime()})\n')

    # Step 0: Set up workspace
    import pandas as pd
    import numpy as np
    import itertools
    import sqlite3
    import fuzzywuzzy
    import re
    from tqdm import tqdm

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
        for i in tqdm(range(len(val_cols))):
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
        for i in tqdm(range(len(val_cols))):
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
        for i in tqdm(range(len(val_cols))):
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

    # Create database connection
    from sqlalchemy import create_engine
    host = '208.43.250.18'
    port = '51949'
    user = 'sa'
    password = 'Aviana$92821'
    driver = '/usr/local/Cellar/freetds/1.1.11/lib/libtdsodbc.0.so'
    # cnxn_str = f'mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver={driver}'
    cnxn_str = f'mssql+pyodbc://@{dsn}/{database}'

    # Make database connection engine
    engine = create_engine(
        cnxn_str,
        # fast_executemany=True,
        # echo=True,
        # implicit_returning=False,
        # isolation_level="AUTOCOMMIT",
    )

    # SQLite for testing and development
    con = sqlite3.connect('revenewML.db')

    # Read in all Cross Reference Files TODO make this a SQL server table
    print('\nReading in supplier cross-reference files...')
    commodity_df = (
        pd.merge(
            pd.read_sql('select Supplier, Commodity from commodities', con),
            pd.read_sql('select Supplier, Supplier_ref from crossref', con),
            on=['Supplier'], how='left'
        ).groupby(['Supplier_ref', 'Commodity']).size().reset_index(name='Freq')
    )[['Supplier_ref', 'Commodity']]
    # print(commodity_df.sample(5))

    # # Read in the new client data (Supplier, Total_Invoice_Amount, Total_Invoice_Count, Year, Client)
    try:
        if database is not None:
            db_tbl = f'{database}.dbo.invoice'
            qry = f'select ProjectID as Client, ' \
                  f'[Vendor Name] as Supplier, ' \
                  f'[Gross Invoice Amount] as Gross_Invoice_Amount ' \
                  f'from {db_tbl}'
            input_df = pd.read_sql(qry, engine)
        elif filename is not None:
            input_df = pd.read_csv(filename,  encoding='ISO-8859-1', sep='\t')
            input_df['Supplier'] = input_df['Supplier'].astype(str)
            input_df['Client'] = input_df['Client'].astype(str)
            # print(input_df.sample(5))
    except:
        input_df = pd.DataFrame()

    # # Merge and find direct matches
    print('\nMatching supplier names against cross-reference file...')
    supplier_crossref_list = pd.read_sql('select Supplier, Supplier_ref from crossref', con)
    input_df_with_ref = pd.merge(input_df, supplier_crossref_list, on='Supplier', how='left')[[
        'Supplier', 'Total_Invoice_Amount', 'Total_Invoice_Count', 'Year', 'Client', 'Supplier_ref'
    ]]
    matched = input_df_with_ref[input_df_with_ref['Supplier_ref'].notnull()]
    # print(matched.sample(5))

    # # Find suppliers not matched with the crossref file
    unmatched = input_df_with_ref[input_df_with_ref['Supplier_ref'].isnull()][['Supplier']]
    unmatched = unmatched.rename(columns={'Supplier': 'Unmatched_Supplier'}).reset_index()
    unmatched = unmatched.groupby(['Unmatched_Supplier']).size().reset_index(name='Freq')
    # print(unmatched)

    # # Try to softmatch the unmatched
    print('\nCleaning up supplier names...')
    unmatched['key'] = 1
    supplier_crossref_list['key'] = 1
    unmatched_cross_ref = pd.merge(unmatched, supplier_crossref_list, on='key').drop('key', axis=1)
    unmatched_supplier = unmatched_cross_ref.Unmatched_Supplier
    unmatched_cross_ref['Unmatched_Supplier_Cleaned'] = [clean_up_string(s) for s in unmatched_supplier]

    print('\nFinding supplier soft-matches...')
    name1 = 'Unmatched_Supplier_Cleaned'
    name2 = 'Supplier_ref'
    unmatched_cross_ref['Unmatched_Supplier_Cleaned'] = unmatched_cross_ref['Unmatched_Supplier_Cleaned'].astype(str)
    unmatched_cross_ref['Supplier_ref'] = unmatched_cross_ref['Supplier_ref'].astype(str)
    iterable = zip(unmatched_cross_ref[name1], unmatched_cross_ref[name2])
    unmatched_cross_ref['MatchRatio'] = [fuzz_ratio(a, b) for a, b in iterable]

    best_matches = sel_distinct(unmatched_cross_ref, ['Unmatched_Supplier'], ['MatchRatio'], 1)[
        ['Unmatched_Supplier', 'Supplier_ref', 'MatchRatio']]
    soft_matches = best_matches.loc[best_matches['MatchRatio'] > 90]
    soft_matches = soft_matches[['Unmatched_Supplier', 'Supplier_ref']].rename(
        columns={'Unmatched_Supplier': 'Supplier'})
    input_df_with_ref.update(soft_matches)
    # print(soft_matches)

    no_soft_matches = best_matches.loc[best_matches['MatchRatio'] <= 90][['Unmatched_Supplier']]
    print(f'\nNo supplier matches found:\n\n{no_soft_matches}')

    input_df_with_ref = input_df_with_ref.rename(columns={'Total_Invoice_Amount_Sum': 'Total_Invoice_Amount'})
    input_df_with_ref = input_df_with_ref.rename(columns={'Total_Invoice_Count_Sum': 'Total_Invoice_Count'})
    input_df_with_ref['Avg_Invoice_Size'] = input_df_with_ref['Total_Invoice_Amount'] / input_df_with_ref[
        'Total_Invoice_Count']
    # print(input_df_with_ref.sample(5))

    # # Bring in the commodity
    print('\nGrouping suppliers by commodity...')
    input_df_with_ref = pd.merge(input_df_with_ref, commodity_df, on=['Supplier_ref'], how='left')
    input_df_with_ref["Commodity"].fillna("NOT_AVAILABLE", inplace=True)
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
    # print(input_df_with_ref.sample(5))

    # Read in the scorecard
    print('\nCalculating supplier scores based on scorecard...')
    supplier_scorecard = pd.read_sql('select * from scorecard', con)
    # print(supplier_scorecard)

    # Do a full outer join with the scorecard
    input_df_with_ref['key'] = 1
    supplier_scorecard['key'] = 1
    input_df_with_ref_with_scorecard = pd.merge(input_df_with_ref, supplier_scorecard, on='key').drop('key', axis=1)
    # print(input_df_with_ref_with_scorecard.sample(5))

    # Get the Spend sub-dataframe
    print('\nCalculating supplier spend score...')
    input_df_with_ref_with_scorecard_spend = (
        input_df_with_ref_with_scorecard.loc[input_df_with_ref_with_scorecard['Factor'] == 'Spend']).reset_index()
    input_df_with_ref_with_scorecard_spend = (
        input_df_with_ref_with_scorecard_spend.loc[input_df_with_ref_with_scorecard_spend['Total_Invoice_Amount']
                                                   < input_df_with_ref_with_scorecard_spend['Max']])
    input_df_with_ref_with_scorecard_spend = (
        input_df_with_ref_with_scorecard_spend.loc[input_df_with_ref_with_scorecard_spend['Total_Invoice_Amount']
                                                   > input_df_with_ref_with_scorecard_spend['Min']])
    input_df_with_ref_with_scorecard_spend = input_df_with_ref_with_scorecard_spend.reset_index()
    # print(input_df_with_ref_with_scorecard_spend.sample(5))

    # Get the InvoiceSize sub-dataframe
    print('\nCalculating invoice size score...')
    input_df_with_ref_with_scorecard_invoicesize = (
        input_df_with_ref_with_scorecard.loc[input_df_with_ref_with_scorecard['Factor'] == 'InvoiceSize']).reset_index()
    input_df_with_ref_with_scorecard_invoicesize = (
        input_df_with_ref_with_scorecard_invoicesize.loc[
            input_df_with_ref_with_scorecard_invoicesize['Avg_Invoice_Size']
            < input_df_with_ref_with_scorecard_invoicesize['Max']])
    input_df_with_ref_with_scorecard_invoicesize = (
        input_df_with_ref_with_scorecard_invoicesize.loc[
            input_df_with_ref_with_scorecard_invoicesize['Avg_Invoice_Size']
            > input_df_with_ref_with_scorecard_invoicesize['Min']])
    input_df_with_ref_with_scorecard_invoicesize = input_df_with_ref_with_scorecard_invoicesize.reset_index()
    # print(input_df_with_ref_with_scorecard_invoicesize.sample(5))

    # Get the InvoiceCount sub-dataframe
    print('\nCalculating invoice count score...')
    input_df_with_ref_with_scorecard_invoicecount = (
        input_df_with_ref_with_scorecard.loc[
            input_df_with_ref_with_scorecard['Factor'] == 'InvoiceCount']).reset_index()
    input_df_with_ref_with_scorecard_invoicecount = (
        input_df_with_ref_with_scorecard_invoicecount.loc[
            input_df_with_ref_with_scorecard_invoicecount['Total_Invoice_Count']
            < input_df_with_ref_with_scorecard_invoicecount['Max']])
    input_df_with_ref_with_scorecard_invoicecount = (
        input_df_with_ref_with_scorecard_invoicecount.loc[
            input_df_with_ref_with_scorecard_invoicecount['Total_Invoice_Count']
            > input_df_with_ref_with_scorecard_invoicecount['Min']])
    input_df_with_ref_with_scorecard_invoicecount = input_df_with_ref_with_scorecard_invoicecount.reset_index()
    # print(input_df_with_ref_with_scorecard_invoicecount.sample(5))

    # Get the Commodity sub-dataframe
    print('\nCalculating commodity score...')
    input_df_with_ref_with_scorecard_commodity = (
        input_df_with_ref_with_scorecard.loc[input_df_with_ref_with_scorecard['Factor'] == 'Commodity']).reset_index()
    input_df_with_ref_with_scorecard_commodity = (
        input_df_with_ref_with_scorecard_commodity.loc[input_df_with_ref_with_scorecard_commodity['Commodity']
                                                       == input_df_with_ref_with_scorecard_commodity['Tier']])
    input_df_with_ref_with_scorecard_commodity = input_df_with_ref_with_scorecard_commodity.reset_index()
    # print(input_df_with_ref_with_scorecard_commodity.sample(5))

    # Append all the factor scores
    print('\nCombining factor scores...')
    scores = (
        input_df_with_ref_with_scorecard_spend.append(input_df_with_ref_with_scorecard_invoicesize, ignore_index=True)
            .append(input_df_with_ref_with_scorecard_invoicecount, ignore_index=True)
            .append(input_df_with_ref_with_scorecard_commodity, ignore_index=True))
    scores = (group_by_stats_list_sum(scores,
                                      ['Client', 'Supplier_ref', 'Year'],
                                      ['Points']
                                      )[['Client', 'Supplier_ref', 'Year', 'Points_Sum']])

    print(f'\nHere are the factor scores:\n\n{scores}')

    # Stop timer
    end = timer()
    elapsed = end - start
    logging.info(
        '\nApplication finished in {:.2f} seconds ... ({})\n'
        '\n<=============================================================>\n\n'.format(
            elapsed, time.ctime()))


if __name__ == '__main__':
    main()
