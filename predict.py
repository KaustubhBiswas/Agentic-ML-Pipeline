import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import re
import argparse
from typing import List, Tuple
from dateutil.parser import parse as date_parse
import os


KNOWN_DATE_FORMATS = [
    'DD-MM-YYYY', 'Dotted_DD.MM.YYYY', 'YYYYMMDD', 'ISO_YYYY-MM-DD', 'MM/DD/YYYY',
    'ISO_Timestamp_Z', 'ISO_Timestamp_Offset', 'Ordinal_Day', 'Weekday_Prefix_Abbrev',
    'TwoDigitYear_DD-MM-YY', 'Weekday_Prefix_Full', 'Mon_DD_YYYY_no_comma', 'Date_With_Time',
    'Full_Month_DD_YYYY', 'Mon_DD,_YYYY', 'Month_Year'
]

KNOWN_PHONE_FORMATS = [
    'Separated by Dashes', 'Plain Digits', 'With Dots', 'Parentheses Variations',
    'Separated by Spaces', 'International (E.164)', 'With Country Code + Spaces/Dashes',
    'Extension Numbers', 'National (local style)', 'Short Codes'
]


def is_likely_date(value: str) -> bool:
    try:
        date_parse(value, fuzzy=False)
        return True
    except:
        return False

def is_likely_phone(value: str) -> bool:
    return bool(re.match(r'^[\+\(\)\d\s\-\.x#]{5,}$', value))

def is_likely_company(value: str) -> bool:
    return len(value.split()) > 1 and any(c.isupper() for c in value) \
        and not is_likely_date(value) and not is_likely_phone(value)


def extract_features(value: str) -> dict:
    return {
        'length': len(value),
        'digit_ratio': sum(c.isdigit() for c in value) / len(value) if len(value) > 0 else 0,
        'alpha_ratio': sum(c.isalpha() for c in value) / len(value) if len(value) > 0 else 0,
        'special_char_count': sum(not c.isalnum() and c not in ['+', '-', '.', '(', ')', ' ', 'x', '#'] for c in value),
        'has_plus': int('+' in value),
        'has_dash': int('-' in value),
        'has_slash': int('/' in value),
        'has_dot': int('.' in value),
        'has_parentheses': int('(' in value or ')' in value),
        'has_extension': int('ext' in value.lower() or 'x' in value.lower()),
        'word_count': len(value.split())
    }


def prepare_training_data():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "TestingData")

    phone_df = pd.read_csv(os.path.join(data_dir, "phone.csv"))
    company_df = pd.read_csv(os.path.join(data_dir, "company.csv"))
    countries_df = pd.read_csv(os.path.join(data_dir, "countries.txt"), header=None)
    dates_df = pd.read_csv(os.path.join(data_dir, "dates.csv"))

    # Phone
    phone_data_raw = phone_df.iloc[:, 0].dropna().astype(str).tolist()
    if len(phone_df.columns) > 1:
        format_series = phone_df.iloc[:, 1]
        phone_data = []
        for idx, p in enumerate(phone_data_raw):
            format_val = format_series.iloc[idx]
            if is_likely_phone(p) and (pd.isna(format_val) or format_val in KNOWN_PHONE_FORMATS):
                phone_data.append(p)
    else:
        phone_data = [p for p in phone_data_raw if is_likely_phone(p)]

    # Company
    company_data_raw = company_df.iloc[:, 0].dropna().astype(str).tolist()
    company_data = [c for c in company_data_raw if is_likely_company(c)]

    # Country
    countries = countries_df.iloc[:, 0].dropna().astype(str).tolist()

    # Dates
    dates_data_raw = dates_df.iloc[:, 0].dropna().astype(str).tolist()
    if len(dates_df.columns) > 1:
        format_series = dates_df.iloc[:, 1]
        dates_data = []
        for idx, d in enumerate(dates_data_raw):
            format_val = format_series.iloc[idx]
            if is_likely_date(d) and (pd.isna(format_val) or format_val in KNOWN_DATE_FORMATS):
                dates_data.append(d)
    else:
        dates_data = [d for d in dates_data_raw if is_likely_date(d)]

    # Other/noise
    other_data = []
    other_data += [p for p in phone_data_raw if p not in phone_data]
    other_data += [d for d in dates_data_raw if d not in dates_data]
    other_data += [c for c in company_data_raw if c not in company_data]
    other_data = list(set(other_data))

    data = phone_data + company_data + countries + dates_data + other_data
    labels = (
        [0] * len(phone_data) +
        [1] * len(company_data) +
        [2] * len(countries) +
        [3] * len(dates_data) +
        [4] * len(other_data)
    )

    return data, labels


def train_model():
    data, labels = prepare_training_data()
    feature_data = [extract_features(v) for v in data]

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)
    X_tfidf = vectorizer.fit_transform(data)

    features_df = pd.DataFrame(feature_data)
    X_dense = features_df.values

    X = hstack((X_tfidf, X_dense))
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.2f}")

    return model, vectorizer


def classify_values(values: List[str], model, vectorizer) -> Tuple[str, float]:
    if not values:
        return "Other", 0.0

    X_tfidf_new = vectorizer.transform(values)
    features_new = [extract_features(v) for v in values]
    features_df_new = pd.DataFrame(features_new)
    X_dense_new = features_df_new.values
    X_new = hstack((X_tfidf_new, X_dense_new))

    predictions = model.predict(X_new)
    proba = model.predict_proba(X_new)

    from collections import Counter
    label_counts = Counter(predictions)
    majority_label = label_counts.most_common(1)[0][0]
    mean_prob = proba[:, majority_label].mean()

    label_map = {0: "Phone Number", 1: "Company Name", 2: "Country", 3: "Date", 4: "Other"}
    return label_map[majority_label], mean_prob

def classify_value(value: str, model, vectorizer) -> Tuple[str, float]:
    if not value:
        return "Other", 0.0

    X_tfidf = vectorizer.transform([value])
    features = [extract_features(value)]
    features_df = pd.DataFrame(features)
    X_dense = features_df.values
    X_new = hstack((X_tfidf, X_dense))

    prediction = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0][prediction]

    label_map = {0: "Phone Number", 1: "Company Name", 2: "Country", 3: "Date", 4: "Other"}
    return label_map[prediction], float(proba)


if __name__ == "__main__":
    model, vectorizer = train_model()
    parser = argparse.ArgumentParser(description='Classify semantic type of a column.')
    parser.add_argument('--input', type=str, required=True, help='Path to input file')
    parser.add_argument('--column', type=str, required=True, help='Column name to classify')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    values = df[args.column].dropna().astype(str).tolist()
    result, prob = classify_values(values, model, vectorizer)
    print(result)
    print(f"Confidence: {prob:.2f}")
