import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
    
    def clean_data(self):
        # Remove duplicates
        self.data.drop_duplicates(inplace=True)
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        columns_to_impute = ['study_hours_per_week', 'attendance_percentage']
        self.data[columns_to_impute] = imputer.fit_transform(self.data[columns_to_impute])
        
        return self
    
    def split_features_target(self):
        # Select features and target
        features = [
            'previous_sem_cgpa', 
            'attendance_percentage', 
            'extracurricular_activities', 
            'study_hours_per_week', 
            'backlogs_previous_sem', 
            'internship_experience'
        ]
        
        X = self.data[features]
        y = self.data['next_sem_cgpa']
        
        return X, y
    
    def prepare_data(self):
        # Clean data
        self.clean_data()
        
        # Split features and target
        X, y = self.split_features_target()
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler