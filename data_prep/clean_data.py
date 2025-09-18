import pandas as pd
import numpy as np

# TODO: Still need to handle dual-major students. Right now I just process them twice.
# Maybe that's actually desirable, but need to think it through.

def load_data():
    df = pd.read_csv('../data_organized/Course history/REGR_HODGE R DATA BUILD Report_20240209.csv')
    return df

def clean_data(df):
    # Keep only relevant columns
    cols = ['SUBJECT', 'COURSE_NUMBER', 'COURSE_OFFERING_SEMESTER', 'STUDENT_ID', 'FINAL_GRADE', 'STUDENT_COLLEGE',
            'STUDENT_CLASSIFICATION_DESC', 'MAJOR_DESC', 'COURSE_COLLEGE']
    df = df[cols]

    # Create course column
    df['COURSE'] = df['SUBJECT'] + ' ' + df['COURSE_NUMBER']

    # Filter out summer semesters
    df = df[~df['COURSE_OFFERING_SEMESTER'].str.contains('Summer')]

    # Create an ordered ID for each semester
    semesters = [
        'Fall 2014', 'Spring 2015',
        'Fall 2015', 'Spring 2016',
        'Fall 2016', 'Spring 2017',
        'Fall 2017', 'Spring 2018',
        'Fall 2018', 'Spring 2019',
        'Fall 2019', 'Spring 2020',
        'Fall 2020', 'Spring 2021',
        'Fall 2021', 'Spring 2022',
        'Fall 2022', 'Spring 2023',
        'Fall 2023'
    ]

    assert set(semesters) == set(df['COURSE_OFFERING_SEMESTER'].unique())

    semester_map = dict(zip(semesters, range(len(semesters))))

    df['SEMESTER_ID'] = df['COURSE_OFFERING_SEMESTER'].map(semester_map)

    # Filter to students who took at least 10 courses
    df = df.groupby('STUDENT_ID').filter(lambda x: len(x) >= 10)

    # Filter to students who took no more than 60 courses
    df = df.groupby('STUDENT_ID').filter(lambda x: len(x) <= 60)

    # Filter to students who took at least two semesters
    df = df.groupby('STUDENT_ID').filter(lambda x: len(x['SEMESTER_ID'].unique()) > 1)

    # Group by student and create a dense rank based on the semester
    df['SEMESTER_RANK'] = df.groupby('STUDENT_ID')['SEMESTER_ID'].rank(method='dense').astype(int)

    # For courses taken less than 100 times, change to "<OTHER>"
    course_counts = df['COURSE'].value_counts()
    df['COURSE'] = df['COURSE'].apply(lambda x: '<OTHER>' if x in course_counts[course_counts < 100].index else x)

    # Assign a unique course ID
    course_id_map, id_course_map = course_id_maps(df)
    df['COURSE_ID'] = df['COURSE'].map(course_id_map)

    # For majors with less than 500 people, change to "<OTHER>"
    major_counts = df['MAJOR_DESC'].value_counts()
    df['MAJOR_DESC'] = df['MAJOR_DESC'].apply(lambda x: '<OTHER>' if x in major_counts[major_counts < 500].index else x)
    df.loc[df['MAJOR_DESC'].isna(), 'MAJOR_DESC'] = '<OTHER>'

    # Assign a unique major ID
    major_id_map, id_major_map = major_id_maps(df)
    df['MAJOR_ID'] = df['MAJOR_DESC'].map(major_id_map)

    # Assign GPA to grades
    # TODO: This doesn't properly account for W, WA, P, NP, etc. Need to consider how to handle those.
    df['GPA'] = 0.0

    df.loc[df['FINAL_GRADE'].str.startswith('A'), 'GPA'] = 4.0
    df.loc[df['FINAL_GRADE'].str.startswith('A+'), 'GPA'] = 4.3
    df.loc[df['FINAL_GRADE'].str.startswith('A-'), 'GPA'] = 3.7

    df.loc[df['FINAL_GRADE'].str.startswith('B'), 'GPA'] = 3.0
    df.loc[df['FINAL_GRADE'].str.startswith('B+'), 'GPA'] = 3.3
    df.loc[df['FINAL_GRADE'].str.startswith('B-'), 'GPA'] = 2.7

    df.loc[df['FINAL_GRADE'].str.startswith('C'), 'GPA'] = 2.0
    df.loc[df['FINAL_GRADE'].str.startswith('C+'), 'GPA'] = 2.3
    df.loc[df['FINAL_GRADE'].str.startswith('C-'), 'GPA'] = 1.7

    df.loc[df['FINAL_GRADE'].str.startswith('D'), 'GPA'] = 1.0
    df.loc[df['FINAL_GRADE'].str.startswith('D+'), 'GPA'] = 1.3
    df.loc[df['FINAL_GRADE'].str.startswith('D-'), 'GPA'] = 0.7

    df.loc[df['FINAL_GRADE'].str.startswith('P'), 'GPA'] = 2.0

    # Keep the final grade if students repeat the course in the same semester (not common, but it happens)
    df = df.drop_duplicates(subset=['STUDENT_ID', 'COURSE', 'SEMESTER_ID'])

    return df[['STUDENT_ID', 'COURSE', 'COURSE_ID', 'SEMESTER_RANK', 'MAJOR_ID', 'MAJOR_DESC', 'GPA']]

def course_id_maps(df):
    course_id_map = dict(zip(df['COURSE'].unique(), range(1, len(df['COURSE'].unique())+1)))
    course_id_map['<PAD>'] = 0
    last_course_id = np.max(list(course_id_map.values()))
    course_id_map['<SOS>'] = last_course_id + 1
    course_id_map['<EOS>'] = last_course_id + 2

    id_course_map = {v: k for k, v in course_id_map.items()}

    return course_id_map, id_course_map

def major_id_maps(df):
    # Zero is used as a padding value, so start at 1
    major_id_map = dict(zip(df['MAJOR_DESC'].unique(), range(1, len(df['MAJOR_DESC'].unique())+1)))
    major_id_map['<UNK>'] = 0

    id_major_map = {v: k for k, v in major_id_map.items()}

    return major_id_map, id_major_map