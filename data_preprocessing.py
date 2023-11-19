import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from the specified file path.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)

def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning and preprocessing on the input DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    # Remove unnecessary columns
    columns_to_drop = ['Timestamp', 'Age', 'state', 'Unnamed: 0', 'comments']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Map countries to continents
    country_mapping = {
        "United States": 0, "United Kingdom": 1, "Canada": 0,
        "Russia": 1, "Germany": 1, "Ireland": 1, "Australia": 2,
        "Netherlands": 1, "New Zealand": 2, "Poland": 1, "Italy": 1,
        "South Africa": 3, "Switzerland": 1, "Sweden": 1, "India": 3,
        "France": 1, "Singapore": 3, "Belgium": 1, "Brazil": 0, "Israel": 3,
        "Bulgaria": 1, "Denmark": 1, "Finland": 1, "Mexico": 0,
        "Colombia": 0, "Croatia": 1, "Thailand": 3, "Georgia": 1,
        "Moldova": 1, "China": 3, "Czech Republic": 1, "Austria": 1,
        "Japan": 3, "Hungary": 1, "Bosnia and Herzegovina": 1,
        "Slovenia": 1, "Portugal": 1, "Philippines": 3
    }
    df["Country"] = df["Country"].map(country_mapping)

    # Convert categorical variables to numbers
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
    df['treatment'] = df['treatment'].map({'No': 0, 'Yes': 1})

    # Handle missing values
    df = df.dropna()
    return df
