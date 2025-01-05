import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Make sure you have downloaded the required data for VADER
# Uncomment these lines if you have not already downloaded them
nltk.download('vader_lexicon')
nltk.download('punkt')

def add_name_polarity_scores(train_df, test_df):
    """
    Add sentiment polarity scores to 'train_df' and 'test_df' based on the 'name' column.

    Args:
        train_df (pd.DataFrame): The training DataFrame, must contain a 'name' column.
        test_df (pd.DataFrame): The testing DataFrame, must contain a 'name' column.

    Returns:
        tuple:
            - train_df (pd.DataFrame): Updated training DataFrame with new column 'name_polarity_scores'.
            - test_df (pd.DataFrame): Updated testing DataFrame with new column 'name_polarity_scores'.
    """
    sid = SentimentIntensityAnalyzer()

    train_df['name_polarity_scores'] = train_df['name'].apply(
        lambda x: None if pd.isna(x) else sid.polarity_scores(x)['compound']
    )
    test_df['name_polarity_scores'] = test_df['name'].apply(
        lambda x: None if pd.isna(x) else sid.polarity_scores(x)['compound']
    )

    return train_df, test_df


def add_month_of_last_review(train_df, test_df):
    """
    Extract the month from 'last_review' in both 'train_df' and 'test_df'
    and map numeric months to their names (e.g., '01' -> 'January').

    Args:
        train_df (pd.DataFrame): The training DataFrame, must contain a 'last_review' column.
        test_df (pd.DataFrame): The testing DataFrame, must contain a 'last_review' column.

    Returns:
        tuple:
            - train_df (pd.DataFrame): Updated training DataFrame with new column 'month_of_last_review'.
            - test_df (pd.DataFrame): Updated testing DataFrame with new column 'month_of_last_review'.
    """
    num_to_month_map = {
        '01': "January", '02': "February", '03': "March", '04': "April",
        '05': "May", '06': "June",    '07': "July",   '08': "August",
        '09': "September", '10': "October", '11': "November", '12': "December"
    }

    train_df['month_of_last_review'] = train_df['last_review'].apply(
        lambda x: x if pd.isna(x) else x.split('-')[1]
    )
    train_df['month_of_last_review'] = train_df['month_of_last_review'].apply(
        lambda x: x if pd.isna(x) else num_to_month_map[x]
    )

    test_df['month_of_last_review'] = test_df['last_review'].apply(
        lambda x: x if pd.isna(x) else x.split('-')[1]
    )
    test_df['month_of_last_review'] = test_df['month_of_last_review'].apply(
        lambda x: x if pd.isna(x) else num_to_month_map[x]
    )

    return train_df, test_df
