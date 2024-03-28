import torch
import random

def mask_video_frames(video_tensor, mask_num):

    total_frames = video_tensor.shape[2]  # batch, channel, frame, H, W
    mask_num = min(mask_num, total_frames)

    frames_to_extract = random.sample(range(total_frames), k=mask_num) 
    remaining_frames_indices = [i for i in range(total_frames) if i not in frames_to_extract]
    extracted_frames_indices = frames_to_extract

    device = video_tensor.device
    video_A = torch.zeros(1, 3, total_frames, 224, 224, device=device)
    video_B = torch.zeros(1, 3, total_frames, 224, 224, device=device)

    for new_index, old_index in enumerate(remaining_frames_indices):
        video_A[:, :, new_index, :, :] = video_tensor[:, :, old_index, :, :]

    for new_index, old_index in enumerate(extracted_frames_indices):
        video_B[:, :, new_index, :, :] = video_tensor[:, :, old_index, :, :]

    return video_A, remaining_frames_indices, video_B, extracted_frames_indices

def restore_masked_frames(batch_new_videos, batch_new_list):
    device = batch_new_videos.device
    rearranged_videos = torch.zeros_like(batch_new_videos,device=device)

    for i, video in enumerate(batch_new_videos):
        if i == len(batch_new_videos) // 2:
            rearranged_videos[i] = video
            continue

        current_list = batch_new_list[i if i < len(batch_new_videos) // 2 else i - 1]   
        for new_idx, old_idx in enumerate(current_list):
            rearranged_videos[i,:,old_idx,:,:] = video[:, new_idx, :, :]
    return rearranged_videos
