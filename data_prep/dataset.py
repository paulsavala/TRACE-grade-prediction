import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from . import utils


# TODO: Make a proper start token, and correctly handle padded GPAs (not padding 0.0, which is the same as failing)
PADDING_VALUE = 0
PADDING_VALUE_DECIMAL = 0.0


def course_semester_tensors(df):
    """
    Description.

    Keyword arguments:
    df -- DataFrame, the input DataFrame containing the student-course data
    """
    # Create a sequence of courses for each student, ignoring the semester
    student_seqs = dict()
    # Store the course ids for each student
    student_tensors = []
    # Store the semesters for each course id for each student. student_semesters and$
    # student_tensors are ordered the same, so that course i in student_tensors matches
    # semester i in student_semesters.
    student_semesters = []
    # Store the student major
    student_majors = []
    # Store the grades in each course in each semester
    student_grades = []

    for id, group in df.groupby('STUDENT_ID'):
        student_seqs[id] = group['COURSE_ID'].values
        student_tensors.append(torch.LongTensor(group['COURSE_ID'].values))
        student_semesters.append(torch.LongTensor(group['SEMESTER_RANK'].values))
        student_majors.append(torch.LongTensor([group['MAJOR_ID'].values[0]]))
        # Normalize GPA to lie between 0 and 1
        student_grades.append(torch.Tensor(group['GPA'].values / 4.3))
    
    return student_seqs, student_tensors, student_semesters, student_majors, student_grades


class CourseSequenceDataset(Dataset):
    def __init__(self, student_input_tensors_padded, student_semester_tensors_padded, student_target_tensors, student_semester_target_padded, student_majors, student_input_grades, student_target_grades):
        self.student_input_tensors_padded = student_input_tensors_padded
        self.student_semester_tensors_padded = student_semester_tensors_padded
        self.student_target_tensors = student_target_tensors
        self.student_semester_target_padded = student_semester_target_padded
        self.student_majors = student_majors
        self.student_input_grades = student_input_grades
        self.student_target_grades = student_target_grades

    def __len__(self):
        return len(self.student_target_tensors)
    
    def __getitem__(self, idx):
        return (self.student_input_tensors_padded[idx], 
                self.student_semester_tensors_padded[idx], 
                self.student_target_tensors[idx], 
                self.student_semester_target_padded[idx],
                self.student_majors[idx],
                self.student_input_grades[idx],
                self.student_target_grades[idx])


def course_seq_dataloader(student_tensors, semester_tensors, major_tensors, grade_tensors, n_courses, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, batch_size=32):
    # Store input and target tensors and semesters
    all_student_input_tensors = []
    all_student_semester_tensors = []
    all_student_target_tensors = []
    all_student_target_semester_tensors = []
    all_student_majors = []
    all_student_input_grades_tensors = []
    all_student_target_grades_tensors = []
    all_student_ids = []

    # Note that we already extend the id_course_map to include the start token at the end,
    # so the SOS token is the final element of that map.
    SOS_TOKEN_DECIMAL = 0.0
    EOS_TOKEN_DECIMAL = 0.0

    # Create inputs and targets, iterating over students
    for i, student in enumerate(student_tensors):
        # Start with their major
        major = torch.LongTensor(major_tensors[i])

        # For each student, grab their semesters and iterate over them
        for semester in semester_tensors[i].unique(sorted=True):
            # Append the student ID (to keep trach of which row is which student)
            all_student_ids.append(i)
            
            # Grab the grades and courses for this semester
            grades = grade_tensors[i]
            student_semesters = semester_tensors[i]

            # Append their major (we need it every semester for the student for indexing purposes),
            # but we only use it for their first semester.
            all_student_majors.append(major)

            # Find the indices for the current (target) and past (input) semesters
            current_semester_idxs = (student_semesters == semester).nonzero()
            prior_semester_idxs = (student_semesters < semester).nonzero()
            
            # If this is the first semester, their major is the input and their courses
            # are the target. Just fill with the padding value. Concatenating their major happens
            # in the model forward pass (after embedding it).
            if semester.item() == 1:
                # Make an empty tensor for the input and target with the appropriate shape
                input = torch.empty((0, 1), dtype=torch.long) 
                input_grades = torch.empty((0, 1), dtype=torch.float) 
                input_semesters = torch.empty((0, 1), dtype=torch.long)
            else:
                # Create the target and input, prepending the major to the input
                input = student[prior_semester_idxs]
                input_grades = grades[prior_semester_idxs]
                input_semesters = student_semesters[prior_semester_idxs]

            # Sort both the courses and grades in the same order, according to the course
            # number. This is done to aid with prediction so that the model can have some understanding of
            # sequencing. TODO: Look into other ways to accomplish this that may be more natural.
            input, input_grades = utils.sort_by_semester(input, input_grades, input_semesters)

            # Prepend a start token
            input = torch.cat((torch.full((1,1), SOS_TOKEN), input), dim=0)
            input_grades = torch.cat((torch.full((1,1), SOS_TOKEN_DECIMAL), input_grades), dim=0)

            # Append an end token
            input = torch.cat((input, torch.full((1,1), EOS_TOKEN)), dim=0)
            input_grades = torch.cat((input_grades, torch.full((1,1), EOS_TOKEN_DECIMAL)), dim=0)

            # Prepend the first semester as a start token
            input_semesters = torch.cat((torch.full((1,1), 1), input_semesters), dim=0)

            # Append the last semester as an end token
            input_semesters = torch.cat((input_semesters, torch.full((1,1), semester-1)), dim=0)

            # Target is always classes for the current semester. Note that this _does not_ include
            # their major, even if it's their first semester (major is always used only as input).
            target = student[current_semester_idxs]
            target_grades = grades[current_semester_idxs]

            # Sort both the courses and grades in the same order, according to the course
            # number. This is done to aid with prediction so that the model can have some understanding of
            # sequencing. TODO: Look into other ways to accomplish this that may be more natural.
            target, target_grades = utils.sort_by_semester(target, target_grades, torch.ones_like(target))
            
            # Prepend a start token
            target = torch.cat((torch.full((1,1), SOS_TOKEN), target), dim=0)
            target_grades = torch.cat((torch.full((1,1), SOS_TOKEN_DECIMAL), target_grades), dim=0)

            # Append an end token
            target = torch.cat((target, torch.full((1,1), EOS_TOKEN)), dim=0)
            target_grades = torch.cat((target_grades, torch.full((1,1), EOS_TOKEN_DECIMAL)), dim=0)

            # Create the target semesters (sequence order), prepending first semester for the major
            target_semesters = student_semesters[current_semester_idxs]

            # Prepend the current semester as a start token
            target_semesters = torch.cat((torch.full((1,1), semester), target_semesters), dim=0)

            # Append the current semester as an end token
            target_semesters = torch.cat((target_semesters, torch.full((1,1), semester)), dim=0)

            # Save them all
            all_student_input_tensors.append(input)
            all_student_input_grades_tensors.append(input_grades)
            all_student_semester_tensors.append(input_semesters)
            all_student_target_tensors.append(target.type(torch.LongTensor))
            all_student_target_grades_tensors.append(target_grades)
            all_student_target_semester_tensors.append(target_semesters)

    # Pad inputs, targets, semesters and grades
    all_student_input_tensors_padded = torch.transpose(pad_sequence(all_student_input_tensors, padding_value=PAD_TOKEN), 0, 1)
    all_student_semester_tensors_padded = torch.transpose(pad_sequence(all_student_semester_tensors, padding_value=PAD_TOKEN), 0, 1)
    all_student_target_tensors_padded = torch.transpose(pad_sequence(all_student_target_tensors, padding_value=PAD_TOKEN), 0, 1)
    all_student_target_semester_tensors_padded = torch.transpose(pad_sequence(all_student_target_semester_tensors, padding_value=PAD_TOKEN), 0, 1)
    all_student_input_grades_tensors_padded = torch.transpose(pad_sequence(all_student_input_grades_tensors, padding_value=PADDING_VALUE_DECIMAL), 0, 1)
    all_student_target_grades_tensors_padded = torch.transpose(pad_sequence(all_student_target_grades_tensors, padding_value=PADDING_VALUE_DECIMAL), 0, 1)

    # Create padding masks for targets (0 for non-masked, 1 for masked)
    tgt_courses_key_padding_mask = torch.zeros_like(all_student_target_tensors_padded)
    for i, y in enumerate(all_student_target_tensors_padded):
        y_padding_idxs = (y.squeeze() == PAD_TOKEN).nonzero()
        tgt_courses_key_padding_mask[i, y_padding_idxs.squeeze()] = torch.ones_like(y_padding_idxs)
    tgt_courses_key_padding_mask = tgt_courses_key_padding_mask.type(torch.BoolTensor)

    # Convert major tensors list to a tensor of the appropriate size
    all_student_major_tensors = torch.stack(all_student_majors, dim=0)

    # Train-test split the dataset
    num_students = len(set(all_student_ids))
    train_size = int(0.9 * num_students)

    # Randomly sample train_size values from the list of student ids
    shuffled_ids = torch.randperm(num_students).tolist()
    train_student_ids = shuffled_ids[:train_size]
    test_student_ids = shuffled_ids[train_size:]

    # Get indices
    train_indices = [i for i, x in enumerate(all_student_ids) if x in train_student_ids]
    test_indices = [i for i, x in enumerate(all_student_ids) if x in test_student_ids]

    assert set(train_indices).intersection(set(test_indices)) == set(), 'Overlap between training and testing sets!'

    # Filter the student tensors based on the train and test student ids
    train_student_input_tensors_padded = all_student_input_tensors_padded[train_indices]
    train_student_semester_tensors_padded = all_student_semester_tensors_padded[train_indices]
    train_student_target_tensors_padded = all_student_target_tensors_padded[train_indices]
    train_student_target_semester_tensors_padded = all_student_target_semester_tensors_padded[train_indices]
    train_student_major_tensors = all_student_major_tensors[train_indices]
    train_student_input_grades_tensors_padded = all_student_input_grades_tensors_padded[train_indices]
    train_student_target_grades_tensors_padded = all_student_target_grades_tensors_padded[train_indices]

    test_student_input_tensors_padded = all_student_input_tensors_padded[test_indices]
    test_student_semester_tensors_padded = all_student_semester_tensors_padded[test_indices]
    test_student_target_tensors_padded = all_student_target_tensors_padded[test_indices]
    test_student_target_semester_tensors_padded = all_student_target_semester_tensors_padded[test_indices]
    test_student_major_tensors = all_student_major_tensors[test_indices]
    test_student_input_grades_tensors_padded = all_student_input_grades_tensors_padded[test_indices]
    test_student_target_grades_tensors_padded = all_student_target_grades_tensors_padded[test_indices]

    # Create the datasets
    train_dataset = CourseSequenceDataset(train_student_input_tensors_padded.squeeze(),
                                        train_student_semester_tensors_padded.squeeze(),
                                        train_student_target_tensors_padded.squeeze(),
                                        train_student_target_semester_tensors_padded.squeeze(),
                                        train_student_major_tensors.squeeze(),
                                        train_student_input_grades_tensors_padded.squeeze(),
                                        train_student_target_grades_tensors_padded.squeeze())
    
    test_dataset = CourseSequenceDataset(test_student_input_tensors_padded.squeeze(),
                                        test_student_semester_tensors_padded.squeeze(),
                                        test_student_target_tensors_padded.squeeze(),
                                        test_student_target_semester_tensors_padded.squeeze(),
                                        test_student_major_tensors.squeeze(),
                                        test_student_input_grades_tensors_padded.squeeze(),
                                        test_student_target_grades_tensors_padded.squeeze())


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.T
    return mask

def generate_src_padding_masks(src_courses, major, pad_idx):
    # Create padding masks for inputs (0 for non-masked, 1 for masked)
    src_device = src_courses.device.type
    src_major_key_padding_mask = torch.zeros_like(major).bool().unsqueeze(1).to(src_device)
    src_courses_key_padding_mask = (src_courses == pad_idx).type(torch.BoolTensor).to(src_device)
    src_gpas_key_padding_mask = src_courses_key_padding_mask
    src_key_padding_mask = torch.cat((src_major_key_padding_mask, src_courses_key_padding_mask, src_gpas_key_padding_mask), dim=1)

    return src_key_padding_mask

def generate_tgt_padding_masks(tgt_courses, pad_idx):
    # Create padding masks for targets (0 for non-masked, 1 for masked)
    tgt_device = tgt_courses.device.type
    tgt_courses_key_padding_mask = (tgt_courses == pad_idx).to(tgt_device)
    tgt_gpas_key_padding_mask = tgt_courses_key_padding_mask
    tgt_key_padding_mask = torch.cat((tgt_courses_key_padding_mask, tgt_gpas_key_padding_mask), dim=1)

    return tgt_key_padding_mask

def generate_src_mask(src_seq_len):
    src_mask = torch.zeros((src_seq_len, src_seq_len)).float()
    return src_mask

def generate_tgt_mask(tgt_seq_len):
    tgt_courses_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_gpas_mask = generate_square_subsequent_mask(tgt_seq_len)
    tgt_mask = torch.cat((tgt_courses_mask, tgt_gpas_mask), dim=1)
    tgt_mask = torch.cat((tgt_mask, tgt_mask), dim=0)
    return tgt_mask