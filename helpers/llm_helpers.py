import pandas as pd

def get_schema_text_for_llm(df):
    """
    Gives a string that describes all the numbers and categories in your DataFrame.
    
    Inputs:
        df (DataFrame):  dataset with numerical and text columns.
        
    Outputs::
        str: A readable summary with:
            - Stats for all the numerical columns (mean, mode, median, kurtosis, etc)
            - Stats about the categorical columns (mode, missing %, mode, etc)
    """
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    num_summary = df[num_cols].describe().T
    num_summary['missing_pct'] = df[num_cols].isna().mean() * 100
    num_summary['n_unique'] = df[num_cols].nunique()
    num_summary['skew'] = df[num_cols].skew()
    num_summary['kurtosis'] = df[num_cols].kurt()
    num_summary['mode'] = df[num_cols].mode().iloc[0]
    
    cat_summary = df[cat_cols].describe().T
    cat_summary['missing_pct'] = df[cat_cols].isna().mean() * 100
    cat_summary['2nd_mode'] = [df[c].value_counts().index[1] if df[c].value_counts().shape[0] > 1 else None for c in cat_cols]
    cat_summary['2nd_mode_freq'] = [df[c].value_counts().iloc[1] if df[c].value_counts().shape[0] > 1 else None for c in cat_cols]
    
    schema_str = (
        "Numerical Variables Summary:\n" +
        num_summary[['count', 'missing_pct', 'n_unique', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis', 'mode']].to_string() +
        "\n\nCategorical Variables Summary:\n" +
        cat_summary[['count', 'missing_pct', 'unique', 'top', 'freq', '2nd_mode', '2nd_mode_freq']].to_string()
    )
    return schema_str

def make_summary(df, informations, n_records=3, eda_findings=None):
    """
    This function summarize a raw dataset for a LLM.

    Inputs:

    df (pandas.DataFrame): The dataset to summarize.
    informations (dict): A dictionary mapping column names to attribute-range/information text.
    n_records (int): Number of sample data rows from the head of the DataFrame to include in the summary, by default its value is 3.
    eda_findings (str or None): By default=None because it's optional. If applicable, this should be wrote by the data scientist with findings of the EDA.
    
    Output:
    
    str:  Info about the size, columns, types, missing values, short description, sample data, stats, and if applicable some EDA notes.
    """
    lines = []
    lines.append(f"Rows: {df.shape[0]}; Columns: {df.shape[1]}\n")
    lines.append("Column Attribute range:")
    for col in df.columns:
        desc = informations.get(col, "")
        lines.append(f"- {col} ({df.dtypes[col]}) | Missing: {df[col].isna().sum()} | Attribute range: {desc}")
    lines.append("\nSample data:\n" + df.head(n_records).to_string())
    lines.append("\nKey statistics (describe):\n" + get_schema_text_for_llm(df))
    if eda_findings:
        lines.append("\n# EDA Highlights:\n" + eda_findings)
    return "\n".join(lines)