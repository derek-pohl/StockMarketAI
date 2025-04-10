import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import warnings
from scipy.stats import randint
from stock_list import stock_data # Import market cap data

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Configuration ---
DATA_FILE = 'stock_training_data.csv'
TARGET_COLUMN = 'Outcome'
DATE_COLUMN = 'Date'
TICKER_COLUMN = 'Ticker'
EARNINGS_FLAG_COLUMN = 'IsEarningsDay'
EARNINGS_TIME_COLUMN = 'EarningsTimeOfDay'

# --- Load Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    # Read CSV, attempt to clean column names immediately
    df = pd.read_csv(DATA_FILE)
    # Clean column names: remove leading/trailing whitespace and extra commas
    df.columns = df.columns.str.strip().str.replace(',$', '', regex=True)
    print("Data loaded successfully.")
    # print("Initial Columns:", df.columns.tolist()) # Debugging
except FileNotFoundError:
    print(f"Error: File not found at {DATA_FILE}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Preprocessing ---
print("Preprocessing data...")
# Convert Date column
try:
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
    df = df.dropna(subset=[DATE_COLUMN]) # Drop rows where date conversion failed
except KeyError:
    print(f"Error: Date column '{DATE_COLUMN}' not found.")
    exit()

# Sort data
df = df.sort_values(by=[TICKER_COLUMN, DATE_COLUMN]).reset_index(drop=True)

# --- Feature Engineering: Target Variable ---
print("Engineering target variable (Outcome)...")
# Calculate next day's close price for each ticker
df['NextDayClose'] = df.groupby(TICKER_COLUMN)['Close'].shift(-1)

# Identify earnings days
earnings_df = df[df[EARNINGS_FLAG_COLUMN] == 1].copy()

# Define Outcome: 1 if NextDayClose > Close on earnings day, 0 otherwise
earnings_df[TARGET_COLUMN] = (earnings_df['NextDayClose'] > earnings_df['Close']).astype(int)

# Drop rows where NextDayClose is NaN (e.g., the very last day for a ticker if it was an earnings day)
earnings_df = earnings_df.dropna(subset=['NextDayClose', TARGET_COLUMN])

if earnings_df.empty:
    print("Error: No valid earnings events found after calculating the target variable.")
    exit()

# --- Prepare Data for Cross-Validation ---
print("Preparing data for Time Series Cross-Validation...")

# Use all valid earnings events
# Ensure data is sorted chronologically for TimeSeriesSplit
earnings_df = earnings_df.sort_values(by=DATE_COLUMN)

# Define features (X) and target (y)
features = ['Open', 'High', 'Low', 'Close', 'InsiderBuyCount', 'InsiderSellCount', 'InsiderBuyShares', 'InsiderSellShares', EARNINGS_TIME_COLUMN]
numerical_features = ['Open', 'High', 'Low', 'Close', 'InsiderBuyCount', 'InsiderSellCount', 'InsiderBuyShares', 'InsiderSellShares']
categorical_features = [EARNINGS_TIME_COLUMN]

# Check if all feature columns exist
missing_cols = [col for col in features if col not in earnings_df.columns]
if missing_cols:
    print(f"Error: The following feature columns are missing from the data: {missing_cols}")
    exit()

X = earnings_df[features]
y = earnings_df[TARGET_COLUMN]

# Handle potential NaN in categorical features before encoding
X[EARNINGS_TIME_COLUMN] = X[EARNINGS_TIME_COLUMN].fillna('None')

# --- Preprocessing Pipeline (Scaling & Encoding) ---
print("Setting up preprocessing pipeline...")


# Create the preprocessing transformer
# Use handle_unknown='ignore' for OneHotEncoder in case test split has categories not seen in train split
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # sparse=False might be needed for some classifiers if not using pipeline
    ],
    remainder='passthrough'
)

# --- Process Market Cap Data ---
print("Processing market cap data from stock_list.py...")
market_cap_lookup = {}
for ticker, data in stock_data.items():
    try:
        # Assuming market cap is the 3rd element (index 2)
        if len(data) >= 3 and isinstance(data[2], (int, float)):
            market_cap_lookup[ticker] = float(data[2])
        else:
            # print(f"Warning: Invalid or missing market cap data for {ticker} in stock_list.py")
            pass # Silently ignore tickers with bad data for now
    except Exception as e:
        # print(f"Warning: Error processing market cap for {ticker}: {e}")
        pass # Silently ignore

print(f"Processed market cap data for {len(market_cap_lookup)} tickers.")

# --- Filter DataFrame by Market Cap ---
print("Filtering earnings data for stocks with 500M <= Market Cap < 1B...")
# Map market cap to the earnings_df
earnings_df['MarketCap'] = earnings_df[TICKER_COLUMN].map(market_cap_lookup)
# Apply the filter
initial_rows = len(earnings_df)
earnings_df_filtered = earnings_df[
    earnings_df['MarketCap'].between(500_000_000, 1_000_000_000, inclusive='left')
].copy() # >= 500M and < 1B
filtered_rows = len(earnings_df_filtered)
print(f"Filtered data from {initial_rows} to {filtered_rows} earnings events.")

if earnings_df_filtered.empty:
    print("Error: No earnings events found within the specified market cap range.")
    exit()

# --- Use Filtered Data for X and y ---
X = earnings_df_filtered[features]
y = earnings_df_filtered[TARGET_COLUMN]

# --- Define Model Pipeline ---
# Base model - hyperparameters will be tuned
rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')

# Full pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', rf_classifier)])

# --- Hyperparameter Tuning Setup ---
print("\n--- Hyperparameter Tuning with RandomizedSearchCV ---")

# Define the parameter distribution to sample from
param_dist = {
    'classifier__n_estimators': randint(100, 500), # Number of trees
    'classifier__max_depth': [5, 10, 20, 30, None], # Max depth of trees (None means full depth)
    'classifier__min_samples_split': randint(2, 11), # Min samples to split a node
    'classifier__min_samples_leaf': randint(1, 6), # Min samples per leaf node
    'classifier__max_features': ['sqrt', 'log2', None] # Number of features to consider at each split
}

# Setup TimeSeriesSplit for cross-validation within RandomizedSearchCV
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Setup RandomizedSearchCV
n_iter_search = 30 # Number of parameter settings that are sampled. Increase for potentially better results, but longer runtime.
random_search = RandomizedSearchCV(
    model_pipeline,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    cv=tscv,
    scoring='accuracy', # Evaluate based on accuracy
    random_state=42,
    n_jobs=-1, # Use all available CPU cores
    verbose=1 # Show progress
)

# --- Run Tuning on Filtered Data ---
print(f"\nStarting Randomized Search with {n_iter_search} iterations and {n_splits}-fold Time Series CV...")
print("Training and Testing ONLY on stocks with 500M <= Market Cap < 1B")

# Fit RandomizedSearchCV on the FILTERED historical dataset
random_search.fit(X, y) # X and y are now pre-filtered

# --- Final Results ---
print("\n--- Hyperparameter Tuning Summary (Filtered Data: 500M - 1B Market Cap) ---")
print(f"Best Parameters Found:")
best_params_cleaned = {k.replace('classifier__', ''): v for k, v in random_search.best_params_.items()}
print(best_params_cleaned)

print(f"\nBest Cross-Validated Accuracy on Filtered Data: {random_search.best_score_:.4f}")

# Optional: Display detailed results for each fold if needed
# cv_results = pd.DataFrame(random_search.cv_results_)
# print("\nCV Results Details:")
# print(cv_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])


print("\nScript finished.")
