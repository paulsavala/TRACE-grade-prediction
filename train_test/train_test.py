import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from data_prep.utils import shuffle_by_timestep, shuffle_non_padded, masked_mae, compute_iou
from data_prep import dataset

from matplotlib import pyplot as plt

from scipy.spatial.distance import squareform


def train(device, dataloader, model, course_loss_fn, gpa_loss_fn, optimizer, scheduler, config, EOS_IDX, PAD_IDX):
    size = len(dataloader.dataset)

    model.to(device)

    # Put model in training mode
    model.train()

    # Iterate over batches
    for batch, (src_courses, src_courses_positions, tgt_courses, tgt_courses_positions, majors, src_gpas, target_grades) in enumerate(dataloader):
        # Shift target to the right
        # Input targets to the decoder ignore the last element
        # Prediction targets ignore the first element
        tgt_courses_input = tgt_courses.detach().clone().to(device)
        tgt_grades_input = target_grades.detach().clone().to(device)
        tgt_pos_input = tgt_courses_positions.detach().clone().to(device)

        # Find index of EOS_IDX in tgt_courses
        eos_tgt_loc = (tgt_courses_input == EOS_IDX).nonzero()
        tgt_courses_input[eos_tgt_loc[:, 0], eos_tgt_loc[:, 1]] = PAD_IDX
        tgt_grades_input[eos_tgt_loc[:, 0], eos_tgt_loc[:, 1]] = 0.
        tgt_pos_input[eos_tgt_loc[:, 0], eos_tgt_loc[:, 1]] = 0

        # Now trim off the last element
        tgt_courses_input = tgt_courses_input[:, :-1].to(device)
        tgt_grades_input = tgt_grades_input[:, :-1].to(device)
        tgt_pos_input = tgt_pos_input[:, :-1].to(device)

        tgt_out = tgt_courses[:, 1:].to(device) 
        tgt_grades_out = target_grades[:, 1:].to(device)
        tgt_pos_out = tgt_courses_positions[:, 1:].to(device)

        # Place on device
        src_courses = src_courses.to(device)
        src_courses_positions = src_courses_positions.to(device)
        majors = majors.to(device)
        src_gpas = src_gpas.to(device)

        # Forward pass
        output_courses_lsm, output_gpas, output_courses = model(major=majors,
                                                                src_courses=src_courses, 
                                                                tgt_courses=tgt_courses_input, 
                                                                src_gpas=src_gpas, 
                                                                tgt_gpas=tgt_grades_input, 
                                                                src_courses_positions=src_courses_positions, 
                                                                tgt_courses_positions=tgt_pos_input)


        # Mask the padding values when computing loss
        course_loss = course_loss_fn(output_courses_lsm, tgt_out)

        # GPA loss function
        # Mask the padding values when computing loss
        gpa_loss = gpa_loss_fn(output_gpas.squeeze(2), tgt_grades_out)
        mask = torch.ones_like(tgt_grades_out)
        mask[tgt_pos_out == 0] = 0
        masked_loss = gpa_loss * mask
        gpa_reduced_loss = masked_loss.sum() / mask.sum()

        loss = config['course_loss_weight']*course_loss + gpa_reduced_loss

        # Backprop
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]['lr'] 

        if batch % 200 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(src_courses)
            course_weight_pct = config['course_loss_weight']*course_loss / loss
            gpa_weight_pct = gpa_reduced_loss / loss
            print(f'loss: {loss_val:>7f} [{current/size:.1%}] ({course_weight_pct:.1%} course, {gpa_weight_pct:.1%} gpa)')
    print(f'LR = {current_lr:.2E}')
    scheduler.step()
    return loss.item()

def test(device, dataloader, model, course_loss_fn, gpa_loss_fn, config, EOS_IDX, PAD_IDX, id_course_map=None):
    # Put the model in evaluation mode
    model.eval()

    iou_list = []

    # Make predictions but don't calculate gradients on forward pass
    with torch.no_grad():
        # Iterate over batches
        for batch, (src_courses, src_courses_positions, tgt_courses, tgt_courses_positions, majors, src_gpas, target_grades) in enumerate(dataloader):
            # Shift target to the right
            # Input targets to the decoder ignore the last element
            # Prediction targets ignore the first element
            tgt_courses_input = tgt_courses.detach().clone().to(device)
            tgt_grades_input = target_grades.detach().clone().to(device)
            tgt_pos_input = tgt_courses_positions.detach().clone().to(device)

            # Find index of EOS_IDX in tgt_courses
            eos_tgt_loc = (tgt_courses_input == EOS_IDX).nonzero()
            tgt_courses_input[eos_tgt_loc[:, 0], eos_tgt_loc[:, 1]] = PAD_IDX
            tgt_grades_input[eos_tgt_loc[:, 0], eos_tgt_loc[:, 1]] = 0.
            tgt_pos_input[eos_tgt_loc[:, 0], eos_tgt_loc[:, 1]] = 0

            # Now trim off the last element
            tgt_courses_input = tgt_courses_input[:, :-1].to(device)
            tgt_grades_input = tgt_grades_input[:, :-1].to(device)
            tgt_pos_input = tgt_pos_input[:, :-1].to(device)

            tgt_out = tgt_courses[:, 1:].to(device) 
            tgt_grades_out = target_grades[:, 1:].to(device)
            tgt_pos_out = tgt_courses_positions[:, 1:].to(device)

            # Place on device
            src_courses = src_courses.to(device)
            src_courses_positions = src_courses_positions.to(device)
            majors = majors.to(device)
            src_gpas = src_gpas.to(device)

            # Forward pass
            output_courses_lsm, output_gpas, output_courses = model(major=majors,
                                                                    src_courses=src_courses, 
                                                                    tgt_courses=tgt_courses_input, 
                                                                    src_gpas=src_gpas, 
                                                                    tgt_gpas=tgt_grades_input, 
                                                                    src_courses_positions=src_courses_positions, 
                                                                    tgt_courses_positions=tgt_pos_input)

            # # Course prediction loss
            course_loss = course_loss_fn(output_courses_lsm, tgt_out)

            # GPA loss function
            # Mask the padding values when computing loss
            gpa_loss = gpa_loss_fn(output_gpas.squeeze(2), tgt_grades_out)
            mask = torch.ones_like(tgt_grades_out)
            mask[tgt_pos_out == 0] = 0
            masked_loss = gpa_loss * mask
            gpa_reduced_loss = masked_loss.sum() / mask.sum()

            loss = config['course_loss_weight']*course_loss + gpa_reduced_loss
        
            # Get class predictions
            pred_courses = output_courses.argmax(dim=1)

            # Compute IOU for courses for the whole batch
            mean_iou = compute_iou(pred_courses, tgt_out, PAD_IDX, EOS_IDX, output_courses.size(1), True)
            iou_list.append(mean_iou.item())

            # Get predicted grades
            # gpas_mask = tgt_pos_out != 0

            # Compute MAE
            # mae = masked_mae(output_gpas.squeeze(2), tgt_grades_out, gpas_mask)      
    
    # print(f'Loss: {loss:.4f}, Mean IOU: {mean_iou:.1%}, MAE: {mae:.4f} \n')
    print(f'Sample')
    if id_course_map is not None:
        print(f'\tPredicted: {[id_course_map[x.item()] for x in pred_courses[0].squeeze() if x not in [0, EOS_IDX, PAD_IDX]]}')
        print(f'\tActual: {[id_course_map[x.item()] for x in tgt_out[0].squeeze() if x not in [0, EOS_IDX, PAD_IDX]]}')
    else:
        print(f'\tPredicted: {pred_courses[0].squeeze()}')
        print(f'\tActual: {tgt_out[0].squeeze()}')
    print(f'Test loss: {loss.item():>7f}')
    print(f'Mean IOU: {np.mean(iou_list):.1%}')
    print()

    return loss.item(), np.mean(iou_list)


# function to generate output sequence using greedy algorithm
def greedy_decode(device, model, src_courses, src_gpas, major, src_mask, src_courses_positions, src_key_padding_mask, SOS_IDX, PAD_IDX=0, max_len=15):
    model.eval()

    batch_size = 1

    # Get the max value from each row
    current_semester = src_courses_positions.max(dim=1).values

    # Encode to get memory
    memory = model.encode(major, 
                          src_courses, 
                          src_gpas, 
                          src_courses_positions, 
                          src_mask, 
                          src_key_padding_mask).to(device)
    
    # Start with the SOS token
    courses_seq = torch.ones(batch_size, 1).fill_(SOS_IDX).type(torch.long).to(device)
    gpas_seq = torch.zeros(batch_size, 1).type(torch.float).to(device)
    courses_pos = torch.zeros(batch_size, 1).type(torch.long).to(device)

    for i in range(max_len-1):
        current_len = courses_seq.size(1)
        tgt_mask = dataset.generate_tgt_mask(current_len).to(device)
        tgt_key_padding_mask = dataset.generate_tgt_padding_masks(courses_seq, PAD_IDX).to(device)

        # print(f'courses_seq[0]: {courses_seq[0]}')
        # print(f'tgt_mask[0]: {tgt_mask[0]}')
        # print(f'tgt_key_padding_mask[0]: {tgt_key_padding_mask[0]}')
        # print(f'tgt_mask: {tgt_mask}')
        # print(f'tgt_key_padding_mask: {tgt_key_padding_mask}')

        # Decode memory and last generated token(s)
        # print(f'memory: {memory.size()}')
        # print(f'courses_seq: {courses_seq.size()}')
        # print(f'gpas_seq: {gpas_seq.size()}')
        # print(f'courses_pos: {courses_pos.size()}')
        # print(f'tgt_mask: {tgt_mask.size()}')
        # print(f'current_len: {current_len}')

        out = model.decode(memory, 
                           courses_seq, 
                           gpas_seq, 
                           courses_pos, 
                           tgt_mask, 
                           tgt_key_padding_mask)

        # Take the last token's output
        prob_courses, prob_gpas = model.generator(out, current_len)
        _, next_course = torch.max(prob_courses[:, :, -1], dim=1)
        next_course = next_course.unsqueeze(1)
        # print(f'prob_gpas: {prob_gpas.size()}')
        next_gpa = prob_gpas[:, -1, :]

        # print(f'courses_seq: {courses_seq.size()}')
        # print(f'prob_courses: {prob_courses.size()}')
        # print(f'prob_courses: {prob_courses.squeeze()}')
        # print(f'gpas_seq: {gpas_seq.size()}')
        # print(f'prob_gpas: {prob_gpas.size()}')
        # print(f'next_course: {next_course.size()}')
        # print(f'next_gpa: {next_gpa.size()}')
        # print(f'next_course: {next_course.squeeze()}')
        # print(f'prob_gpas: {prob_gpas.squeeze()}')
        # print(f'next_course: {next_course.squeeze()}')
        
        # courses_seq and gpas_seq grows in sequence length dimension
        courses_seq = torch.cat([courses_seq, next_course], dim=1)
        gpas_seq = torch.cat([gpas_seq, next_gpa], dim=1)
        courses_pos = torch.cat([courses_pos, current_semester.unsqueeze(1)], dim=1)
    return courses_seq, gpas_seq

# actual function to translate input sentence into target language
def predict(device, model, major, src_courses, src_gpas, src_courses_positions, SOS_IDX, PAD_IDX=0):
    # Source mask
    src_seq_len = 1 + src_courses.size(1) + src_gpas.size(1)
    src_mask = dataset.generate_src_mask(src_seq_len).to(device)
    
    # Padding masks
    src_key_padding_mask = dataset.generate_src_padding_masks(src_courses, major, PAD_IDX).to(device)

    # Place on device
    major = major.to(device)
    src_courses = src_courses.to(device)
    src_gpas = src_gpas.to(device)
    src_courses_positions = src_courses_positions.to(device)
    src_key_padding_mask = src_key_padding_mask.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        pred_courses, pred_gpas = greedy_decode(device,
                                                model, 
                                                src_courses, 
                                                src_gpas,
                                                major,
                                                src_mask,
                                                src_courses_positions,
                                                src_key_padding_mask,
                                                SOS_IDX,
                                                PAD_IDX)
    return pred_courses, pred_gpas