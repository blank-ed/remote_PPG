from utils import *
from filters import *
from mne.preprocessing import ICA

def apply_jade(signal, n_components=3, random_state=None):
    ica = ICA(n_components=n_components, method='fastica', random_state=random_state)
    ica.fit(signal)
    sources = ica.get_sources(signal).T
    second_component = sources[:, 1]
    return second_component

raw_sig = VJ_face_detector(r"C:\Users\Admin\Desktop\Riccardo New Dataset\test_L00_no_ex_riccardo_all_distances\D01.mp4")
windowed_sig = moving_window(raw_sig)

for y, x in enumerate(windowed_sig):
    normalized = normalize(x)
    # ICA = apply_jade(normalized)
