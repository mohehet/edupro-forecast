import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def run_pipeline():
    print("Loading data from Excel...")
    try:
        excel_file = pd.ExcelFile('EduPro Online Platform.xlsx')
        courses = pd.read_excel(excel_file, sheet_name='Courses')
        teachers = pd.read_excel(excel_file, sheet_name='Teachers')
        transactions = pd.read_excel(excel_file, sheet_name='Transactions')
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    print("Merging and aggregating data...")
    # Aggregate transaction data at the Course-Teacher level
    tx_agg = transactions.groupby(['CourseID', 'TeacherID']).agg(
        enrollment_count=('TransactionID', 'count'),
        course_revenue=('Amount', 'sum')
    ).reset_index()
    
    # Merge datasets
    df = tx_agg.merge(courses, on='CourseID', how='left')
    df = df.merge(teachers, on='TeacherID', how='left')
    
    # Handle missing/sparse
    df['enrollment_count'] = df['enrollment_count'].fillna(0)
    df['course_revenue'] = df['course_revenue'].fillna(0)
    df['CourseRating'] = df['CourseRating'].fillna(df['CourseRating'].mean())
    df['TeacherRating'] = df['TeacherRating'].fillna(df['TeacherRating'].mean())
    
    # Feature Engineering
    print("Engineering features...")
    # Course features
    def get_price_band(p):
        if p <= 20: return 'Low'
        elif p <= 100: return 'Medium'
        else: return 'High'
    df['price_band'] = df['CoursePrice'].apply(get_price_band)
    df['duration_bucket'] = pd.cut(df['CourseDuration'], bins=[0, 10, 25, 50, 100], labels=['Short', 'Medium', 'Long', 'Very Long'])
    df['rating_tier'] = pd.cut(df['CourseRating'], bins=[0, 3.5, 4.5, 5.0], labels=['Low', 'Medium', 'High'])
    
    # Instructor features
    df['experience_bucket'] = pd.cut(df['YearsOfExperience'], bins=[-1, 5, 10, 20, 50], labels=['Junior', 'Mid', 'Senior', 'Expert'])
    df['match_score'] = (df['CourseCategory'] == df['Expertise']).astype(int)
    
    # Historical / Computed
    df['revenue_per_enrollment'] = np.where(df['enrollment_count'] > 0, df['course_revenue'] / df['enrollment_count'], 0)
    
    # Category Revenue
    cat_revenue = df.groupby('CourseCategory')['course_revenue'].sum().reset_index().rename(columns={'course_revenue': 'category_revenue'})
    df = df.merge(cat_revenue, on='CourseCategory', how='left')
    
    # Standardize column names to match app.py
    rename_map = {
        'CourseCategory': 'category',
        'CourseLevel': 'level',
        'CoursePrice': 'price',
        'CourseDuration': 'duration_hours',
        'CourseRating': 'course_rating',
        'YearsOfExperience': 'experience_years',
        'TeacherRating': 'teacher_rating'
    }
    df = df.rename(columns=rename_map)
    
    # Ensure categorical consistency by filling NAs in categoricals
    for cat in ['category', 'level']:
        df[cat] = df[cat].fillna('Unknown')
    
    # Drop IDs and highly correlated/leakage features for modeling
    features_to_drop = ['CourseID', 'TeacherID', 'CourseName', 'CourseType', 'TeacherName', 'Age', 'Gender', 'Expertise']
    df = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    
    # Targets
    target_enrollment = 'enrollment_count'
    target_revenue = 'course_revenue'
    target_cat_rev = 'category_revenue'
    
    # Remove targets from features to prevent leakage in revenue predictions
    X = df.drop(columns=[target_enrollment, target_revenue, target_cat_rev, 'revenue_per_enrollment'])
    
    # Categorical and numerical columns
    cat_cols = ['category', 'level', 'price_band', 'duration_bucket', 'rating_tier', 'experience_bucket']
    num_cols = ['price', 'duration_hours', 'course_rating', 'experience_years', 'teacher_rating', 'match_score']
    
    
    # Phase 2: Model Development
    print("Training models...")
    
    def get_models():
        """Return fresh model instances to avoid object sharing between targets."""
        return {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
    
    results = {}
    best_models = {}
    
    # Train for Enrollment Count
    y_enroll = df[target_enrollment]
    X_train, X_test, y_train, y_test = train_test_split(X, y_enroll, test_size=0.2, random_state=42)
    
    best_r2 = -float('inf')
    for name, model in get_models().items():
        pipe = Pipeline(steps=[('preprocessor', ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        )), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        results[f"{name}_Enrollment"] = {
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'R2': r2
        }
        if r2 > best_r2:
            best_r2 = r2
            best_models['Enrollment'] = pipe
            if hasattr(model, 'feature_importances_'):
                best_models['Enrollment_fi'] = model.feature_importances_
    
    # Train for Course Revenue
    y_rev = df[target_revenue]
    X_train, X_test, y_train, y_test = train_test_split(X, y_rev, test_size=0.2, random_state=42)
    
    best_r2 = -float('inf')
    for name, model in get_models().items():
        pipe = Pipeline(steps=[('preprocessor', ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        )), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        results[f"{name}_Revenue"] = {
            'MAE': mean_absolute_error(y_test, preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'R2': r2
        }
        if r2 > best_r2:
            best_r2 = r2
            best_models['Revenue'] = pipe
            if hasattr(model, 'feature_importances_'):
                best_models['Revenue_fi'] = model.feature_importances_
                
    # Save models
    print("Saving best models...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_models['Enrollment'], 'models/enrollment_model.joblib')
    joblib.dump(best_models['Revenue'], 'models/revenue_model.joblib')
    
    # Save processed dataframe for the dashboard (for Category Revenue and Explorer)
    df.to_csv('processed_data.csv', index=False)
    
    # Save Feature names for FI
    pipe = best_models['Revenue']
    cat_features = pipe.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(cat_cols)
    all_features = num_cols + list(cat_features)
    joblib.dump(all_features, 'models/feature_names.joblib')
    
    if 'Revenue_fi' in best_models:
        joblib.dump(best_models['Revenue_fi'], 'models/revenue_fi.joblib')
    if 'Enrollment_fi' in best_models:
        joblib.dump(best_models['Enrollment_fi'], 'models/enrollment_fi.joblib')
        
    print("Evaluation Results:")
    for key, val in results.items():
        print(f"{key}: R2={val['R2']:.3f}, MAE={val['MAE']:.2f}, RMSE={val['RMSE']:.2f}")
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
