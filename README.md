# 📦 Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, LabelEncoder,
    StandardScaler, MinMaxScaler,
    PolynomialFeatures
)
from sklearn.model_selection import train_test_split

sns.set(style='whitegrid')

# 1. Load dataset
df = pd.read_csv('your_data.csv')  # Replace with your CSV file
print("Initial shape:", df.shape)
print(df.info())
print(df.isnull().sum().sort_values(ascending=False).head())

# 2. Handle missing values
# Drop cols with >30% missing
high_missing = df.columns[df.isnull().mean() > 0.3]
df.drop(columns=high_missing, inplace=True)

# Impute numeric with median, categorical with mode
num_cols = df.select_dtypes(include=['int64','float64'])
cat_cols = df.select_dtypes(include=['object','category'])

num_imputer = SimpleImputer(strategy='median')
df[num_cols.columns] = num_imputer.fit_transform(num_cols)

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols.columns] = cat_imputer.fit_transform(cat_cols)

# 3. Detect & cap outliers via IQR
for col in num_cols.columns:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df[col] = df[col].clip(lower, upper)

# 4. Encode categorical features
# Label encode low-cardinality
for col in cat_cols.columns:
    if df[col].nunique() <= 10:
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# 5. Feature Scaling (standard + min-max)
scaler_std = StandardScaler()
scaler_mm = MinMaxScaler()

scaled_cols = num_cols.columns
df[scaled_cols] = scaler_std.fit_transform(df[scaled_cols])
df[scaled_cols] = scaler_mm.fit_transform(df[scaled_cols])

# 6. Feature Engineering
# Example: create interaction terms, polynomial features, datetime extraction
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

# Interaction & polynomial features for numeric
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_feats = poly.fit_transform(df[scaled_cols])
poly_cols = poly.get_feature_names_out(scaled_cols)
df_poly = pd.DataFrame(poly_feats, columns=poly_cols, index=df.index)
df = pd.concat([df, df_poly.drop(columns=scaled_cols)], axis=1)

# 7. Split into train/test
target = 'target'  # replace with your target column name
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found")
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train/Test shapes:", X_train.shape, X_test.shape)

# 8. (Optional) Visual EDA
sns.boxplot(data=pd.DataFrame(X_train[scaled_cols]))
plt.title('Box plot of scaled numeric features')
plt.show()

sns.heatmap(X_train[scaled_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation matrix')
plt.show()

print("✅ Preprocessing & feature engineering complete!")
