

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
    cleaned = cleaned.replace('"', '')
    return cleaned
