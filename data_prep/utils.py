import torch
import torch.nn.functional as F

import numpy as np

# Function to shuffle only values in the tensors based on timesteps.
# We leave padding values untouched. The point is to shuffle non-padding values for
# data augmentation purposes. We also respect the timesteps, so that only values with
# the same timestep are shuffled together.
# Example use case: Courses taken each semester, and we want data augmentation to
# shuffle the courses taken in each semester. However, we _do not_ want courses shuffled
# _between_ semesters, nor do we want padding values to be touched.
def shuffle_by_timestep(tensor1, tensor2, timestep_tensor, padding_value=0):
    assert tensor1.size() == tensor2.size() == timestep_tensor.size(), "All tensors must have the same size"
    
    shuffled_tensor1 = tensor1.clone()
    shuffled_tensor2 = tensor2.clone()
    
    for i in range(tensor1.size(0)):
        timesteps = timestep_tensor[i].unique()
        for t in timesteps:
            if t == padding_value:
                continue
            # Get indices of the current timestep
            timestep_indices = (timestep_tensor[i] == t).nonzero(as_tuple=True)[0]
            
            # Extract non-padded values
            non_padded_indices = timestep_indices[tensor1[i][timestep_indices] != padding_value]
            non_padded_values1 = tensor1[i][non_padded_indices]
            non_padded_values2 = tensor2[i][non_padded_indices]
            
            # Generate a random permutation
            perm = torch.randperm(non_padded_values1.size(0))
            
            # Apply the permutation
            shuffled_tensor1[i][non_padded_indices] = non_padded_values1[perm]
            shuffled_tensor2[i][non_padded_indices] = non_padded_values2[perm]
    
    return shuffled_tensor1, shuffled_tensor2

# A simpler version of the above function where there are no timesteps. Otherwise
# it has the same functionality as the above function. I could probably combine
# these two at some point.
def shuffle_non_padded(tensor1, tensor2, sos_token, eos_token, padding_value=0):
    assert tensor1.size() == tensor2.size(), "Tensors must have the same size"
    
    shuffled_tensor1 = tensor1.clone()
    shuffled_tensor2 = tensor2.clone()
    
    for i in range(tensor1.size(0)):
        # Get indices of non-padded values
        mask = (tensor1[i] != padding_value) & (tensor1[i] != sos_token)  & (tensor1[i] != eos_token)
        non_padded_indices = mask.nonzero(as_tuple=True)[0]
        
        # Extract non-padded values
        non_padded_values1 = tensor1[i][non_padded_indices]
        non_padded_values2 = tensor2[i][non_padded_indices]
        
        # Generate a random permutation
        perm = torch.randperm(non_padded_values1.size(0))
        
        # Apply the permutation
        shuffled_tensor1[i][non_padded_indices] = non_padded_values1[perm]
        shuffled_tensor2[i][non_padded_indices] = non_padded_values2[perm]
    
    return shuffled_tensor1, shuffled_tensor2

def masked_mae(tensor1, tensor2, mask, row_wise=False):
    """
    Compute the masked mean absolute error (MAE) between two tensors.
    
    Args:
    - tensor1: The first tensor of shape [batch_size, ...].
    - tensor2: The second tensor of shape [batch_size, ...].
    - mask: A mask tensor of the same shape as tensor1 and tensor2.
    - row_wise: If True, compute MAE for each row (batch-wise).
               If False, compute MAE across the entire batch.

    Returns:
    - MAE value(s) either for each row or for the entire batch.
    """
    # Step 1: Compute the absolute differences
    abs_diff = torch.abs(tensor1 - tensor2)

    # Step 2: Apply the mask
    masked_diff = abs_diff * mask

    if row_wise:
        # Step 3 (row-wise): Compute the sum and count per row
        masked_sum = masked_diff.sum(dim=1)
        valid_count = mask.sum(dim=1)
        
        # Step 4 (row-wise): Compute row-wise MAE, avoiding division by zero
        mae = masked_sum / valid_count.clamp(min=1)
        return mae
    else:
        # Step 3 (batch-wise): Compute the sum and count across the entire batch
        masked_sum = masked_diff.sum()
        valid_count = mask.sum()
        
        # Step 4 (batch-wise): Compute overall MAE
        mae = masked_sum / valid_count.clamp(min=1)
        return mae
    
def compute_iou(outputs, targets, pad_token, eos_token, num_classes, compute_mean=False):
    """
    Computes the IoU row by row for batched outputs and targets, ignoring padding and EOS tokens.
    
    Args:
        outputs (torch.Tensor): The output predictions from the model of shape [batch_size, seq_length].
        targets (torch.Tensor): The ground truth targets of shape [batch_size, seq_length].
        pad_token (int): The token used for padding, which should be ignored in the IoU computation.
        eos_token (int): The token used for the end of sequence, which should also be ignored.

    Returns:
        torch.Tensor: A tensor of IoU scores, one for each row in the batch.
    """
    # Create a mask to ignore pad and EOS tokens
    valid_mask_outputs = (outputs != pad_token) & (outputs != eos_token)
    valid_mask_targets = (targets != pad_token) & (targets != eos_token)

    # Get valid outputs and targets, ignoring padding and EOS tokens
    valid_outputs = outputs * valid_mask_outputs
    valid_targets = targets * valid_mask_targets

    # One-hot encoding of the valid outputs and targets (ignoring duplicates by summing and clamping)
    outputs_one_hot = torch.nn.functional.one_hot(valid_outputs, num_classes=num_classes).sum(dim=1).clamp(0, 1)
    targets_one_hot = torch.nn.functional.one_hot(valid_targets, num_classes=num_classes).sum(dim=1).clamp(0, 1)

    # Compute intersection and union across the batch
    intersection = (outputs_one_hot & targets_one_hot).sum(dim=1).float()
    union = (outputs_one_hot | targets_one_hot).sum(dim=1).float()

    # Handle division by zero: if the union is zero, IoU should be 1.0
    iou = intersection / union
    iou[union == 0] = 1.0  # If both sets are empty, consider IoU to be 1.0
    
    if compute_mean:
        return iou.mean()
    else:
        return iou

def sort_by_semester(courses, grades, semesters):
    # Create a tensor to store the sorted results
    sorted_courses = torch.empty_like(courses)
    sorted_grades = torch.empty_like(grades)

    # Get the unique timesteps
    unique_timesteps = torch.unique(semesters)

    # Sort each group of elements that share the same timestep
    for timestep in unique_timesteps:
        # Get the indices of the current timestep
        mask = semesters == timestep
        ts_indices = mask.nonzero(as_tuple=True)[0]

        # Sort tensor1 within the current timestep and get the sorted indices
        _, sorted_indices = torch.sort(courses[ts_indices].squeeze(1))

        # Reorder tensor1 and tensor2 based on the sorted indices
        sorted_courses[ts_indices] = courses[ts_indices][sorted_indices]
        sorted_grades[ts_indices] = grades[ts_indices][sorted_indices]

    return sorted_courses, sorted_grades