# from importlib import import_module
# from remote_PPG.sig_extraction_utils import *
# from remote_PPG.utils import *
# from remote_PPG.filters import *
from remote_PPG.methods import *
from remote_PPG.hr_estimator import *
import csv
import ast
import os
from sklearn.metrics import mean_absolute_error
import time
from line_profiler import LineProfiler


def big_framework(input_video, sig_extraction_params=None, px_filter=True, windowing_params=None, pre_filtering=None,
                  method='CHROM', post_filtering=None, hr_estimation='stft_estimator', remove_outlier=False):

    # raw_sig = extract_raw_sig(input_video, **sig_extraction_params, pixel_filtering=px_filter)
    # fps = get_fps(input_video)
    raw_sig = input_video
    fps = 30

    sig_windowing = moving_window(raw_sig, fps=fps, **windowing_params)

    pre_filtered_sig = apply_filters(sig_windowing, pre_filtering)

    bvp_module = import_module('remote_PPG.methods')
    bvp_method = getattr(bvp_module, method)
    bvp = bvp_method(pre_filtered_sig, fps, **windowing_params)

    post_filtered_sig = apply_filters(bvp, post_filtering)

    hrES = get_bpm(post_filtered_sig, fps, hr_estimation, remove_outlier=remove_outlier, bpm_type='continuous')

    return hrES


sig_parameters = [{'framework': 'CHROM', 'ROI_type': 'None', 'width': 1, 'height': 1},
                  {'framework': 'POS', 'ROI_type': 'None', 'width': 1, 'height': 1},
                  {'framework': 'ICA', 'ROI_type': 'None', 'width': 0.6, 'height': 1},
                  {'framework': 'GREEN', 'ROI_type': 'ROI_I', 'width': 1, 'height': 1},
                  {'framework': 'GREEN', 'ROI_type': 'ROI_II', 'width': 1, 'height': 1},
                  {'framework': 'GREEN', 'ROI_type': 'ROI_III', 'width': 1, 'height': 1},
                  {'framework': 'GREEN', 'ROI_type': 'ROI_IV', 'width': 1, 'height': 1},
                  {'framework': 'LiCVPR', 'ROI_type': 'None', 'width': 1, 'height': 1}]
px_filter = [True, False]
window_params = {'window_size': 1.6, 'increment': 0.8}
filtering_methods = ['detrending_filter', 'moving_average_filter', 'butterworth_bp_filter', 'fir_bp_filter']
filtering_combinations = get_filtering_combinations(filtering_methods)
hr_estimation = ['stft_estimator', 'fft_estimator', 'welch_estimator']
outlier_removal = [True, False]

from tqdm import tqdm

# base_dir = r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET'
# for sub_folders in os.listdir(base_dir):
#     if sub_folders == 'UBFC2':
#         for folders in tqdm(os.listdir(os.path.join(base_dir, sub_folders))):
#             subjects = os.path.join(base_dir, sub_folders, folders)
#             for each_subject in os.listdir(subjects):
#                 if each_subject.endswith('.avi'):
#                     vid = os.path.join(subjects, each_subject)
#                 elif each_subject.endswith('.txt'):
#                     gt = os.path.join(subjects, each_subject)
#             for_each_framework = []
#             for each_sig in tqdm(sig_parameters):
#                 for_each_px_filter = []
#                 for px_filtering in px_filter:
#                     raw_sig = extract_raw_sig(vid, **each_sig, pixel_filtering=px_filtering)
#                     for_each_px_filter.append(raw_sig)
#                 for_each_framework.append(for_each_px_filter)
#
#             with open('UBFC2_raw_sigs.txt', 'a') as f:
#                 f.write(str(for_each_framework) + "\n")



# from tqdm import tqdm
# import os
#
# sub_folder_name = None
# folder_name = None
# subject_name = None
# each_sig_name = None
# px_filtering_name = None
#
# try:
#     base_dir = r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET'
#
#     for sub_folder_name in os.listdir(base_dir):
#         if sub_folder_name == 'UBFC2':
#             for folder_name in tqdm(os.listdir(os.path.join(base_dir, sub_folder_name))):
#                 subjects = os.path.join(base_dir, sub_folder_name, folder_name)
#                 for subject_name in os.listdir(subjects):
#                     if subject_name.endswith('.avi'):
#                         vid = os.path.join(subjects, subject_name)
#                     elif subject_name.endswith('.txt'):
#                         gt = os.path.join(subjects, subject_name)
#                 for_each_framework = []
#                 for each_sig_name in tqdm(sig_parameters):
#                     for_each_px_filter = []
#                     for px_filtering_name in px_filter:
#                         raw_sig = extract_raw_sig(vid, **each_sig_name, pixel_filtering=px_filtering_name)
#                         for_each_px_filter.append(raw_sig)
#                     for_each_framework.append(for_each_px_filter)
#
#                 with open('UBFC2_raw_sigs.txt', 'a') as f:
#                     f.write(str(for_each_framework) + "\n")
#
#     print("Simulation completed successfully.")
#     os.system('shutdown /s /t 20')
#
# except Exception as e:
#     print(f"An error occurred: {str(e)}")
#     print(f"Error Details: Sub_folder: {sub_folder_name}, Folder: {folder_name}, Subject: {subject_name}, Sig_Parameters: {each_sig_name}, Pixel_Filter: {px_filtering_name}")
#     with open('UBFC2_raw_sigs.txt', 'a') as f:
#         f.write(f"An error occurred: {str(e)}\n")
#         f.write(f"Error Details: Sub_folder: {str(sub_folder_name)}, Folder: {str(folder_name)}, Subject: {str(subject_name)}, Sig_Parameters: {str(each_sig_name)}, Pixel_Filter: {str(px_filtering_name)}\n")
#     os.system('shutdown /s /t 20')

raw_sig = []
with open('UBFC2_raw_sigs.txt', 'r') as f:
    read = f.readlines()
    for x in read:
        sigs = ast.literal_eval(x)
        raw_sig.append(sigs)

def profile_print(func_to_call, *args, **kwargs):
    profiler = LineProfiler()
    profiler.add_function(func_to_call)
    profiler.runcall(func_to_call, *args, **kwargs)
    profiler.print_stats()



start_time = time.time()
i = 0
for enum_sig_params, each_sig_params in enumerate(sig_parameters):
    for enum_px_filter, px_filtering in enumerate(px_filter):
        for pre_filtering_combo in filtering_combinations:
            for post_filtering_combo in filtering_combinations:
                for hr_estimator in hr_estimation:
                    for removing_outlier in outlier_removal:
                        ground_truth_hr = [108.01886792452831, 94.2741935483871, 105.1063829787234, 99.05660377358491, 112.02898550724638, 108.40579710144928, 110.3623188405797, 123.76811594202898, 68.75, 103.26086956521739, 69.92647058823529, 115.8695652173913, 93.07142857142857, 86.78571428571429, 120.72463768115942, 125.8695652173913, 102.68115942028986, 65.97014925373135, 106.54411764705883, 115.0, 98.33333333333333, 109.38775510204081, 98.04347826086956, 78.76811594202898, 105.43478260869566, 116.83823529411765, 116.44927536231884, 103.64285714285714, 119.28571428571429, 57.10144927536232, 109.14285714285714, 84.92753623188406, 86.15942028985508, 101.01449275362319, 95.14285714285714, 99.48529411764706, 82.89855072463769, 110.14492753623189, 97.20588235294117, 105.94202898550725, 90.8955223880597, 87.82608695652173]
                        estimated_hr = []
                        base_dir = r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET\UBFC2'
                        if i == 4823:
                            print([i, each_sig_params, px_filtering, pre_filtering_combo, post_filtering_combo, hr_estimator, removing_outlier])
                        if i > 4823:

                            print([i, each_sig_params, px_filtering, pre_filtering_combo, post_filtering_combo, hr_estimator, removing_outlier])
                            print(raw_sig[enum_sig_params][enum_px_filter])
                            for enum_subject, folder_name in enumerate(os.listdir(base_dir)):
                                subjects = os.path.join(base_dir, folder_name)
                                for subject_name in os.listdir(subjects):
                                    if subject_name.endswith('.avi'):
                                        vid = os.path.join(subjects, subject_name)
                                    elif subject_name.endswith('.txt'):
                                        gt = os.path.join(subjects, subject_name)
                                hrES = big_framework(
                                    # input_video=r'C:\Users\Admin\Desktop\Riccardo New Dataset\test_L00_no_ex_riccardo_all_distances\D01.mp4',
                                    input_video=raw_sig[enum_subject][enum_sig_params][enum_px_filter],
                                    sig_extraction_params=each_sig_params,
                                    px_filter=px_filtering,
                                    windowing_params=window_params,
                                    pre_filtering=pre_filtering_combo,
                                    method='CHROM',
                                    post_filtering=post_filtering_combo,
                                    hr_estimation=hr_estimator,
                                    remove_outlier=removing_outlier)
                                estimated_hr.append(np.mean(hrES))


                            MAE = mean_absolute_error(ground_truth_hr, estimated_hr)
                        i += 1

                        elapsed_time = time.time() - start_time  # Compute elapsed time
                        avg_time_per_iteration = elapsed_time / i  # Compute average time per iteration
                        remaining_iterations = 13824 - i  # Compute remaining iterations
                        est_remaining_time = remaining_iterations * avg_time_per_iteration  # Estimated remaining time
                        if i > 4824:
                            combo = [i, each_sig_params['framework'], px_filtering, pre_filtering_combo, post_filtering_combo, hr_estimator, removing_outlier, MAE]
                            with open('UBFC2_permutations.csv', 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(combo)
                        print(f"{(i / 13824) * 100}% or {i}/13824 or {13824 - i} left or {est_remaining_time/60} minutes left")

                            # print(each_sig_params, px_filtering, pre_filtering_combo, post_filtering_combo, hr_estimator, removing_outlier, enum_subject)
                            # print(enum_sig_params, enum_px_filter, enum_subject)
                            # print(raw_sig[enum_subject][enum_sig_params][enum_px_filter])
                            # print(len(raw_sig[enum_subject][enum_sig_params][enum_px_filter]))
                            # print('')
                            # time.sleep(0.1)
                        # combo = [i, each_sig_params['framework'], px_filtering, pre_filtering_combo, post_filtering_combo, hr_estimator, removing_outlier, np.mean(output)]
                        # with open('test.csv', 'a', newline='') as f:
                        #     writer = csv.writer(f)
                        #     writer.writerow(combo)
                        # print(f"{(i/13824) * 100}% or {i}/13824 or {13824-i} left")



# -------------------- This is for testing -------------------- #

# sig_parameters = {'framework': 'GREEN', 'ROI_type': 'ROI_I', 'width': 1, 'height': 1}
# window_params = {'window_size': 1.6, 'increment': 0.8}
# filtering_methods = ['detrending_filter', 'moving_average_filter', 'butterworth_bp_filter', 'fir_bp_filter']
# filtering_combinations = get_filtering_combinations(filtering_methods)
#
# print(filtering_combinations[0])
#
# output = big_framework(input_video=r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET\UBFC2\subject01\vid.avi',
#                        sig_extraction_params=sig_parameters,
#                        px_filter=True,
#                        windowing_params=window_params,
#                        pre_filtering=filtering_combinations[0],
#                        method='CHROM',
#                        post_filtering=filtering_combinations[0],
#                        hr_estimation='stft_estimator',
#                        remove_outlier=False)
#
# print(np.mean(output))

