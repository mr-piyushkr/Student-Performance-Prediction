import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic student performance data
def generate_student_performance_data(num_students=500):
    data = {
        'student_id': range(1, num_students + 1),
        'previous_sem_cgpa': np.random.uniform(2.0, 4.0, num_students),
        'attendance_percentage': np.random.uniform(60, 100, num_students),
        'extracurricular_activities': np.random.randint(0, 5, num_students),
        'study_hours_per_week': np.random.uniform(10, 50, num_students),
        'backlogs_previous_sem': np.random.randint(0, 4, num_students),
        'internship_experience': np.random.randint(0, 2, num_students),
        'next_sem_cgpa': np.zeros(num_students)
    }
    
    df = pd.DataFrame(data)
    
    # Create a more realistic CGPA prediction model
    df['next_sem_cgpa'] = (
        0.4 * df['previous_sem_cgpa'] + 
        0.2 * (df['attendance_percentage'] / 20) + 
        0.1 * (df['study_hours_per_week'] / 10) - 
        0.3 * df['backlogs_previous_sem'] + 
        0.1 * df['internship_experience'] + 
        np.random.normal(0, 0.5, num_students)
    ).clip(2.0, 4.0)
    
    # Round CGPA to 2 decimal places
    df['next_sem_cgpa'] = df['next_sem_cgpa'].round(2)
    
    return df

# Generate and save the dataset
student_data = generate_student_performance_data()
student_data.to_csv('student_performance.csv', index=False)
print(student_data.head())
print("\nDataset Statistics:")
print(student_data.describe())