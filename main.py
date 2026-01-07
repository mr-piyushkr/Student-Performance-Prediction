from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
import sys
sys.path.append('.')
from data_preprocessing import DataPreprocessor

def main():
    # Step 1: Data Preprocessing
    preprocessor = DataPreprocessor('student_performance.csv')
    X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_data()
    
    # Step 2: Model Training and Evaluation
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    results = trainer.train_and_evaluate()
    
    # Print Results
    print("Model Performance Results:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    
    # Plot Model Performance
    trainer.plot_results()
    
    # Feature Importance
    feature_names = [
        'Previous Sem CGPA', 
        'Attendance Percentage', 
        'Extracurricular Activities', 
        'Study Hours per Week', 
        'Backlogs Previous Sem', 
        'Internship Experience'
    ]
    feature_importance = trainer.feature_importance(feature_names)
    print("\nFeature Importance:")
    print(feature_importance)

if __name__ == '__main__':
    main()