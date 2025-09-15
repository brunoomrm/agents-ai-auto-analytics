import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import chi2_contingency

### Section A 

def show_shape(df):
    """
    Gives the number of rows and columns in one dataframe

    Input:
        df: pandas DataFrame

    Output:
        Prints number of rows and columns
    """
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

def plot_missing_bar(df):
    """
    Barplot of percentage of missing values per column (variables with at least one missing value).

    Input:
        df: pandas DataFrame
    """
   
    missing_pct = (df.isna().sum() / df.shape[0]) * 100
    missing_df = missing_pct[missing_pct > 0].sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=missing_df.values, y=missing_df.index, palette="Blues_r")
    plt.xlabel("% Missing Values", fontweight="bold", fontsize=12)
    plt.ylabel("Variable", fontweight="bold", fontsize=12)
    plt.title("Percentage of Missing Values by Variable", fontweight="bold", fontsize=14)
    plt.xlim(0, missing_df.values.max() + 5)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_missing_heatmap(df):
    """
    Heatmap of missing values, only for columns with at least one missing value.
    Uses blues palette and bold axis/title.

    Input:
        df: pandas DataFrame
    """

    missing_cols = df.columns[df.isna().any()]
    plt.figure(figsize=(len(missing_cols)*0.7+5, 6))
    sns.heatmap(df[missing_cols].isna(), cmap="Blues", cbar=False)
    plt.xlabel("Variables", fontweight="bold", fontsize=12)
    plt.ylabel("Samples", fontweight="bold", fontsize=12)
    plt.title("Missing Value Heatmap (Variables with Missing Data)", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.show()

def summary_table(df):
    """
    Print summary tables for numeric and categorical variables.

    Input:
        df: pandas DataFrame

    Output:
        Displays two summary tables, one for numerical variables and another for categorical variables.
    """
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    num_summary = df[num_cols].describe().T
    num_summary['missing_pct'] = df[num_cols].isna().mean() * 100
    num_summary['n_unique'] = df[num_cols].nunique()
    num_summary['skew'] = df[num_cols].skew()
    num_summary['kurtosis'] = df[num_cols].kurt()
    num_summary['mode'] = df[num_cols].mode().iloc[0]
    print("\nNumerical Variables Summary:")
    display(num_summary[['count', 'missing_pct', 'n_unique', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis', 'mode']])

    cat_summary = df[cat_cols].describe().T
    cat_summary['missing_pct'] = df[cat_cols].isna().mean() * 100
    cat_summary['2nd_mode'] = [df[c].value_counts().index[1] if df[c].value_counts().shape[0] > 1 else None for c in cat_cols]
    cat_summary['2nd_mode_freq'] = [df[c].value_counts().iloc[1] if df[c].value_counts().shape[0] > 1 else None for c in cat_cols]
    print("\nCategorical Variables Summary:")
    display(cat_summary[['count', 'missing_pct', 'unique', 'top', 'freq', '2nd_mode', '2nd_mode_freq']])

### Section B
def plot_countplots(df, max_unique=10):
    """
    Plot countplots for each categorical variable with <= max_unique categories.

    Input:
        df: pandas DataFrame
        max_unique: int (only plot columns with at most this many unique values)
    """
   
    cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if df[col].nunique() <= max_unique]
    for col in cat_cols:
        plt.figure(figsize=(7, 4))
        sns.countplot(data=df, x=col, palette="pastel")
        plt.xlabel(col, fontsize=19, fontweight='bold')
        plt.ylabel("Count", fontsize=10, fontweight='bold')
        plt.title(f"Countplot of {col}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

def plot_unique_bar(df, max_unique=20):
    """
    Barplot showing number of unique values for each categorical variable.

    Input:
        df: pandas DataFrame
        max_unique: int (max # unique to show for clarity)
    Output:
        Displays the barplot (no returned value)
    """
 
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    unique_counts = df[cat_cols].nunique().sort_values(ascending=False)
    unique_counts = unique_counts[unique_counts <= max_unique]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=unique_counts.values, y=unique_counts.index, palette="Blues_r")
    plt.xlabel("Number of Unique Values", fontweight="bold")
    plt.ylabel("Categorical Variable", fontweight="bold")
    plt.title("Unique Value Count per Categorical Variable", fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_numeric_distributions(df, skip_cols=['price'], uniq_thresh=20):
    """
    For each numerical variable.
      - For discrete variables: barplot + boxplot
      - For continuous variables: histogram/KDE + boxplot

    Input:
        df: pandas DataFrame
        skip_cols: list of columns to skip
        uniq_thresh: int, unique value threshold for barplot vs hist
    """
    
    num_cols = [c for c in df.select_dtypes(include=['number']).columns if c not in skip_cols]
    for col in num_cols:
        n_unique = df[col].nunique(dropna=True)
        col_dtype = df[col].dtype
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        if (col_dtype in ['int64', 'Int64'] and n_unique <= uniq_thresh):
            sns.countplot(x=df[col], ax=axes[0], palette='Blues')
            axes[0].set_xlabel(col, fontsize=12, fontweight='bold')
            axes[0].set_ylabel("Count", fontsize=12, fontweight='bold')
            axes[0].set_title(f"Barplot of {col}", fontsize=14, fontweight='bold')
        else:
            sns.histplot(df[col], kde=True, ax=axes[0], color='steelblue')
            axes[0].set_xlabel(col, fontsize=12, fontweight='bold')
            axes[0].set_ylabel("Count", fontsize=12, fontweight='bold')
            axes[0].set_title(f"Distribution of {col}", fontsize=14, fontweight='bold')
        sns.boxplot(x=df[col], ax=axes[1], color='royalblue')
        axes[1].set_xlabel(col, fontsize=12, fontweight='bold')
        axes[1].set_title(f"Boxplot of {col}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

def plot_price_distribution(df):
    """
    Plot price distribution (histogram/KDE) and logarithm price distribution.

    Input:
        df: pandas DataFrame containing 'price' column
    Output:
    """
    
    plt.figure(figsize=(8, 4))
    sns.histplot(df["price"], kde=True, color='navy')
    plt.xlabel("price", fontweight="bold", fontsize=12)
    plt.ylabel("Count", fontweight="bold", fontsize=12)
    plt.title("Distribution of Price", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.histplot(np.log1p(df["price"].dropna()), kde=True, color='deepskyblue')
    plt.xlabel("log(price)", fontweight="bold", fontsize=12)
    plt.ylabel("Count", fontweight="bold", fontsize=12)
    plt.title("Log Distribution of Price", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.show()

### Section C

def plot_correlation_heatmap(df, annot=True):
    """
    Plot a heatmap of Pearson correlations between numeric variables in the DataFrame.
    Input: df: DataFrame, anoot: Boolean
    """
   
    corr = df.select_dtypes(include='number').corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, cmap='Blues', annot=annot, fmt=".2f", mask=mask,
                cbar_kws={'label': 'Correlation'})
    plt.title("Correlation Matrix of Numeric Variables", fontweight='bold', fontsize=16)
    plt.xlabel("Features", fontweight="bold")
    plt.ylabel("Features", fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_pairplot(df, cols):
    """
    Create a pairplot of selected numeric columns.
    Input: 
    df: DataFrame, 
    cols: list of numeric column names
    """
    
    g = sns.pairplot(df[cols], corner=True, diag_kind="kde", plot_kws={'color': 'steelblue'})
    g.figure.suptitle("Pairplot of Selected Numeric Variables", fontweight='bold', fontsize=16, y=1.02)
    plt.show()

def plot_scatter(df, x_col, y_col, hue_col=None):
    """
    Scatterplot between two numeric columns, group by hue if provided.
    Input: df: DataFrame 
    x_col
      y_col: str
    hue_col: str/None
    """
    
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='Blues', alpha=0.7)
    plt.xlabel(x_col, fontsize=12, fontweight='bold')
    plt.ylabel(y_col, fontsize=12, fontweight='bold')
    title = f"Scatterplot of {y_col} vs {x_col}" if not hue_col else f"Scatterplot of {y_col} vs {x_col} by {hue_col}"
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_grouped_boxplot(df, num_col, cat_col):
    """
    Boxplot of a numeric column grouped by a categorical column.
    Input: 
    df: DataFrame
    num_col: str
    cat_col: str
    """
   
    plt.figure(figsize=(8,5))
    sns.boxplot(x=cat_col, y=num_col, data=df, palette="Blues")
    plt.title(f"Boxplot of {num_col} by {cat_col}", fontsize=14, fontweight='bold')
    plt.xlabel(cat_col, fontsize=12, fontweight='bold')
    plt.ylabel(num_col, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_grouped_violinplot(df, num_col, cat_col):
    """
    Violin plot of a numeric column grouped by a categorical column.
    Input: 
    df: DataFrame
    num_col: str
    cat_col: str
    """
 
    plt.figure(figsize=(8,5))
    sns.violinplot(x=cat_col, y=num_col, data=df, palette="Blues")
    plt.title(f"Violin Plot of {num_col} by {cat_col}", fontsize=14, fontweight='bold')
    plt.xlabel(cat_col, fontsize=12, fontweight='bold')
    plt.ylabel(num_col, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

def cramers_v(x, y):
    """
    Calculate Cramér's V statistic for association between two categorical variables.
    Input: x, y: Series
    Output: cramers v which is a number between 0 and 1
    """

    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def plot_cramers_v_heatmap(df):
    """
    Calculate and plot a heatmap of Cramér's V correlation between all categorical variables present in a datframe.
    Input: df: DataFrame
    """
   
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    cramer_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    for col1 in cat_cols:
        for col2 in cat_cols:
            cramer_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
    mask = np.triu(np.ones_like(cramer_matrix, dtype=bool), k=1)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cramer_matrix, annot=True, cmap='Blues', mask=mask)
    plt.title("Cramér's V Association (Lower Triangle)", fontsize=14, fontweight='bold')
    plt.xlabel("Variable", fontsize=12, fontweight='bold')
    plt.ylabel("Variable", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

def categorical_numerical_eta_summary(df):
    """
    For all categorical (object/category dtype) and numeric columns in df,
    calculate eta squared for every pair.
    
    Input: 
        df: pandas DataFrame
    Output: 
        eta_df: DataFrame of all eta squared values
    """

    def eta_squared(df, cat_col, num_col):
        means = df.groupby(cat_col)[num_col].mean()
        counts = df[cat_col].value_counts()
        overall_mean = df[num_col].mean()
        ssw = ((means - overall_mean)**2 * counts).sum()
        sst = ((df[num_col] - overall_mean)**2).sum()
        if sst == 0:
            return 0.0
        return ssw / sst

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    num_cols = df.select_dtypes(include='number').columns

    eta_results = {}
    for cat in cat_cols:
        for num in num_cols:
            try:
                eta = eta_squared(df, cat, num)
                eta_results[(cat, num)] = eta
            except Exception:
                eta_results[(cat, num)] = float('nan')

    eta_df = pd.Series(eta_results).unstack().T

    return eta_df

def plot_eta_squared_heatmap(eta_df):
    """
    Plot a heatmap of eta squared association strengths between categorical and numeric variables.

    Input:
        eta_df: DataFrame (rows: numeric vars, columns: categorical vars)
    
    """

    plt.figure(figsize=(1 + eta_df.shape[1] * 1.1, 1 + eta_df.shape[0] * .60))
    sns.heatmap(eta_df, annot=True, cmap='Blues', fmt=".2f", 
                linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Eta Squared (η²)'})
    plt.xlabel("Categorical Variable", fontweight='bold')
    plt.ylabel("Numeric Variable", fontweight='bold')
    plt.title(r"Eta Squared ($\eta^2$) for Categorical/Numeric Associations", fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

### Section D

def plot_all_boxplots(df, cols=None):
    """
    Plot boxplots for each specified float numeric column 
    (or all numeric columns if cols is None).
    Input: 
    df - DataFrame
    cols - list of numeric columns (default: all)
    """

    if cols is None:
        cols = df.select_dtypes(include='float64').columns
    for col in cols:
        plt.figure(figsize=(6, 1.5))
        sns.boxplot(x=df[col], color='royalblue')
        plt.title(f"Boxplot of {col}", fontweight='bold', fontsize=12)
        plt.xlabel(col, fontweight='bold')
        plt.tight_layout()
        plt.show()

def flag_zscore_outliers(df, cols=None, z_thresh=3):
    """
    Identify outliers using the Z-score method for each numeric column.
    Input: 
    df - Dataframe 
    cols (optional) - list of columns
    z_thresh (float64) - it gives the thresold for z-score method
    Output: 
    outlier_mask (dataframe): DataFrame with the outliers for each column.
    """

    if cols is None:
        cols = df.select_dtypes(include='float64').columns
    outlier_mask = pd.DataFrame(False, index=df.index, columns=cols)
    for col in cols:
        col_zscore = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        outlier_mask[col] = np.abs(col_zscore) > z_thresh
    return outlier_mask

def flag_iqr_outliers(df, cols=None, iqr_thresh=1.5):
    """
    Identify outliers using IQR  for each numeric column.
    Input: 
    df - Dataframe 
    cols (optional) - list of columns
    iqr_thresh (float64) - it gives the thresold for IRQ method
    Output: 
    outlier_mask (dataframe): DataFrame with the outliers for each column.    
    """
    if cols is None:
        cols = df.select_dtypes(include='float64').columns
    outlier_mask = pd.DataFrame(False, index=df.index, columns=cols)
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_thresh * iqr
        upper = q3 + iqr_thresh * iqr
        outlier_mask[col] = (df[col] < lower) | (df[col] > upper)
    return outlier_mask

def outlier_summary(df, method='iqr', thresh=1.5):
    """
    Create a summary DataFrame: n_outliers, percent outliers, min, max per variable.
    Input: 
    df - dDataframe 
    method (str): 'iqr' or 'zscore'
    thresh (float 64): usually 1.5 for IQR and 3 for z-score.
    """
    if method == 'zscore':
        mask = flag_zscore_outliers(df, z_thresh=thresh)
    else:
        mask = flag_iqr_outliers(df, iqr_thresh=thresh)
    summary = []
    n = len(df)
    for col in mask.columns:
        n_outliers = mask[col].sum()
        pct = n_outliers / n * 100
        col_min = df[col].min()
        col_max = df[col].max()
        summary.append([col, n_outliers, f"{pct:.1f}%", col_min, col_max])
    summary_df = pd.DataFrame(summary, columns=['Variable', 'Num Outliers', '% Outliers', 'Min', 'Max'])
    return summary_df
