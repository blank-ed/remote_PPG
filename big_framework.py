from remote_PPG.utils import *
from remote_PPG.filters import *

# def extract_sig(input_video, parameters):
#
#     framework = parameters['ROI extraction framework']
#     ROI_type = parameters['ROI type']
#
#     if framework == 'GREEN':
#
#
#     if framework == 'CHROM':
#         raw_sig = extract_raw_sig(input_video, framework='CHROM', width=1, height=1, pixel_filtering=True)
#     elif framework == 'POS':
#         raw_sig = extract_raw_sig(input_video, framework='POS', width=1, height=1, pixel_filtering=True)
#     elif framework == 'ICA':
#         raw_sig = extract_raw_sig(input_video, framework='ICA', width=0.6, height=1, pixel_filtering=False)
#     elif framework == 'LiCVPR':
#         raw_sig = extract_raw_sig(input_video, framework='LiCVPR', width=1, height=1, pixel_filtering=False)
#     elif framework == 'GREEN':
#         raw_sig = extract_raw_sig(input_video, framework='GREEN', ROI_type=ROI_type, width=1, height=1, pixel_filtering=False)
#     else:
#         assert False, "Please ensure all parameters are correctly passed. "
#
#     return raw_sig

def big_framework(input_video, sig_extraction_type=None, px_filter=True, pre_filtering={}, method='CHROM',
                  post_filtering={}, hr_estimation='welch', outlier_removal=True):

    raw_sig = extract_raw_sig(input_video, framework=sig_extraction_type, width=1, height=1, pixel_filtering=px_filter)



    return raw_sig


big_framework(r'')
