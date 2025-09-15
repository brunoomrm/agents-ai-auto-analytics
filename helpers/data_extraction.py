import re
import pandas as pd

def extract_section_7(names_path):
    """
    Extract lines belonging to section 7 ("Attribute Information") from a .names file.

    Returns: list of lines (str) in section 7
    """
    with open(names_path, 'r') as f:
        lines = f.readlines()

    section_lines = []
    in_section = False
    for line in lines:
        if re.match(r'^\s*7\.\s*Attribute Information', line):
            in_section = True
            continue
        if in_section and re.match(r'^\s*8\. *Missing', line):
            break
        if in_section:
            section_lines.append(line.rstrip())

    return section_lines

def load_automobile_df(data_path, names_path):
    '''
    Function to load the Automobile dataset as a pandas DataFrame, extracting both column names and their descriptions from section 7 of the .names file.

    Input:
        data_path (str): Path to the data file, which contains the data.
        names_path (str): Path to the names file that is needed to have in data frame the columns names.

    Output:
        df (pandas - pd DataFrame): The automobile data with appropriate columns and NaN for missing values
        informations (dict): Dictionary mapping column names to their informations range
    '''
    attr_lines = extract_section_7(names_path)
    
    columns, informations = [], {}
    current_col = None
    for line in attr_lines:
        line = line.strip()
        if not line or line.startswith("--"):
            continue
        m = re.match(r"^(\d+)\.\s*([\w\-]+):\s*(.*)", line)
        if m:
            _, col_name, desc = m.groups()
            current_col = col_name
            columns.append(col_name)
            informations[current_col] = desc.strip()
        elif line and current_col:
            informations[current_col] += " " + line

    df = pd.read_csv(data_path, header=None, names=columns, na_values='?')
    return df, informations