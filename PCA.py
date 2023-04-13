from remote_PPG.utils import *
from remote_PPG.filters import *

raw_sig = VJ_face_detector(r"C:\Users\ilyas\Desktop\VHR\Datasets\Distance vs Light Dataset\test_all_riccardo_distances_L00_NoEx\D01.mp4")
fps = get_fps(r"C:\Users\ilyas\Desktop\VHR\Datasets\Distance vs Light Dataset\test_all_riccardo_distances_L00_NoEx\D01.mp4")
for enum, sg in enumerate(raw_sig):
    if enum == 0:
        filtered = bp_filter(sig=np.array(raw_sig).all(), fps=fps)
        print(filtered)

