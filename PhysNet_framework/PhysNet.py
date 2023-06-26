# TODO: Have to implement training model on a specified video dataset with their corresponding ground truth data
import sklearn.metrics
import torch
from numpy.fft import fftfreq
from scipy.fft import fft
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torch.optim import Adam
from PhysNetED_BMVC import PhysNet_padding_Encoder_Decoder_MAX
from NegPearsonLoss import Neg_Pearson
from remote_PPG.utils import *
import pandas as pd
import matplotlib.pyplot as plt


class PhysNetDatasetBuilder(Dataset):
    def __init__(self, root_dir, dataset=None, training_length=128):

        self.root_dir = root_dir
        self.training_length = training_length
        self.dataset = dataset

        self.video_files = []
        self.gt_files = []

        if self.dataset == 'UBFC1':
            for each_subject in os.listdir(self.root_dir):
                for ground_truth_files in os.listdir(os.path.join(self.root_dir, each_subject)):
                    if ground_truth_files.endswith('.avi'):
                        self.video_files.append(os.path.join(self.root_dir, each_subject, ground_truth_files))
                    elif ground_truth_files.endswith('.xmp'):
                        self.gt_files.append(os.path.join(self.root_dir, each_subject, ground_truth_files))

        elif self.dataset == 'UBFC2':
            for each_subject in os.listdir(self.root_dir):
                for ground_truth_files in os.listdir(os.path.join(self.root_dir, each_subject)):
                    if ground_truth_files.endswith('.avi'):
                        self.video_files.append(os.path.join(self.root_dir, each_subject, ground_truth_files))
                    elif ground_truth_files.endswith('.txt'):
                        self.gt_files.append(os.path.join(self.root_dir, each_subject, ground_truth_files))

        # elif self.dataset == None:


    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        print(f'Processing: {video_file}')
        frames = np.array(extract_raw_sig(video_file, framework='PhysNet', width=1, height=1))

        gt_file = self.gt_files[idx]
        if self.dataset == 'UBFC1':
            gtdata = pd.read_csv(gt_file, header=None)
            gtTrace = gtdata.iloc[:, 3].tolist()
            gtTrace = gtTrace[::2]  # Resample it to be 30 Hz since the ground truth sensor is at 60 Hz sampling rate

            gtTime = (gtdata.iloc[:, 0] / 1000).tolist()
            gtHR = gtdata.iloc[:, 1]


        elif self.dataset == 'UBFC2':
            gtdata = pd.read_csv(gt_file, delimiter='\t', header=None)
            gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']  # Already resampled
            gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
            gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']

        video_data = frames[:self.training_length]
        ground_truth = np.array(gtTrace[0:self.training_length])

        video_tensor = torch.from_numpy(video_data).permute(3, 0, 1, 2).float()
        bvp_tensor = torch.from_numpy(ground_truth).float()

        return video_tensor, bvp_tensor


def detect_peaks(signal, fps):
    # Calculate the power spectrum by taking the absolute square of the FFT
    power_spectrum = np.abs(fft(signal)) ** 2

    # Calculate the corresponding frequencies
    frequencies = fftfreq(len(signal), d=1 / fps)

    # Display only the positive frequencies and corresponding power spectrum
    positive_frequencies = frequencies[:len(frequencies) // 2]  # Take only the first half
    positive_spectrum = power_spectrum[:len(power_spectrum) // 2]  # Take only the first half
    freq_range = (0.8, 2.0)  # Frequency range

    # Find the indices corresponding to the desired frequency range
    start_idx = np.argmax(positive_frequencies >= freq_range[0])
    end_idx = np.argmax(positive_frequencies >= freq_range[1])

    # Extract the frequencies and power spectrum within the desired range
    frequencies_range = positive_frequencies[start_idx:end_idx]
    power_spectrum_range = positive_spectrum[start_idx:end_idx]
    max_idx = np.argmax(power_spectrum_range)
    f_max = frequencies_range[max_idx]
    hr = f_max * 60.0

    return hr


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device} device")

model = PhysNet_padding_Encoder_Decoder_MAX(frames=128).to(device)
loss_function = Neg_Pearson().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

# test = PhysNetDatasetBuilder(r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET\UBFC1', dataset='UBFC1')
# train = PhysNetDatasetBuilder(r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET\UBFC2', dataset='UBFC2')

dataset = PhysNetDatasetBuilder(r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET\UBFC2', dataset='UBFC2')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Start training
model.train()
num_epochs = 15
for epoch in range(num_epochs):
    for video, bvp in train_loader:
        optimizer.zero_grad()

        video, bvp = video.to(device), bvp.to(device)

        rPPG, x_visual, x_visual3232, x_visual1616 = model(video)
        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
        BVP_label = (bvp - torch.mean(bvp)) / torch.std(bvp)  # normalize

        loss = loss_function(rPPG, BVP_label)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')



model.eval()  # Set the model to evaluation mode

# Define the file path where you want to save the model
save_path = r"C:\Users\Admin\PycharmProjects\pythonProject2\remote_PPG\split_80_30"

# Save the model
torch.save(model.state_dict(), save_path)

total_mae = 0
hrES = []
hrGT = []
with torch.no_grad():  # Do not calculate gradients for efficiency
    for video, bvp in test_loader:
        video = video.cuda()  # if using GPU
        bvp = bvp.cuda()  # if using GPU

        # Forward pass
        rPPG, x_visual, x_visual3232, x_visual1616 = model(video)

        rPPG_filtered = butterworth_bp_filter(signal=rPPG.cpu().numpy(), fps=30, low=0.67, high=2.0)
        bvp_filtered = butterworth_bp_filter(signal=bvp.cpu().numpy(), fps=30, low=0.67, high=2.0)

        rPPG_normalized = (np.array(rPPG_filtered) - np.mean(rPPG_filtered)) / np.std(rPPG_filtered)
        bvp_normalized = (np.array(bvp_filtered) - np.mean(bvp_filtered)) / np.std(bvp_filtered)

        print(rPPG_normalized.tolist())
        print(bvp_normalized.tolist())

        for enum, signal in enumerate(rPPG_normalized):
            rPPG_hr = detect_peaks(rPPG_normalized[enum], fps=30)
            bvp_hr = detect_peaks(bvp_normalized[enum], fps=30)

            # print(rPPG_hr, bvp_hr)
            hrES.append(rPPG_hr)
            hrGT.append(bvp_hr)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(hrGT, hrES))

# 20.552884615384617 70 30
# 21.875 80 20

rppg_hr = [[-9.832555819370144e-08, -2.7461168066362025e-08, -2.5088046763481867e-07, -5.315198094398796e-07, -3.071446331517419e-08, 4.274825614069409e-08, -5.20881543541939e-07, -3.530316940376227e-07, 1.948360112997763e-08, -5.658222808128326e-07, -5.915746547924151e-07, -3.0026079856537183e-07, 4.430087096374077e-07, -1.725241197494105e-07, -5.667968135653628e-07, -5.012340104734284e-07, 1.3966338660593468e-07, -2.1257567743687792e-07, -6.584091298646198e-07, -6.001761677559853e-07, 1.2972518319328813e-07, -2.2537832906412023e-07, -8.167928850480458e-07, -6.243754962172442e-07, 1.3369958362019579e-07, -1.9253185862848183e-07, -8.660254792632782e-07, -6.431510415857053e-07, 1.7340288867749533e-07, -1.4535249500878274e-07, -7.088203936829093e-07, -7.020576135679702e-07, 2.811914393559723e-07, -7.99588192895823e-08, -7.920344267619933e-07, -6.537795752469098e-07, 2.721661585607992e-07, -6.20431431507077e-08, -7.825382170217189e-07, -6.491989591956443e-07, 3.2165212824265537e-07, -8.78029952580646e-08, -7.153872929292818e-07, -6.088590885426161e-07, 3.6655419459857783e-07, 3.618586480188674e-08, -6.913993815474972e-07, -6.919834211015135e-07, 3.133997688962381e-07, 5.069704357116681e-10, -7.43492616447398e-07, -7.478628473206123e-07, 3.738109574290032e-07, 8.370795098768607e-08, -6.510452821402342e-07, -7.879286170357314e-07, 3.9194383847504583e-07, 1.8883427727174088e-07, -5.91812189366756e-07, -6.183487632359738e-07, 5.562128672688149e-07, 4.115277086594751e-07, -6.306589250815609e-07, -6.882023172427149e-07, 3.638565988640003e-07, 2.592040759832909e-07, -6.319748204452454e-07, -8.016124666948187e-07, 4.5041437212237646e-07, 1.1201078477054851e-07, -6.560048744048513e-07, -6.729443934388376e-07, 3.8621564836639735e-07, 1.90457709881988e-07, -7.276397929506413e-07, -5.5505604195917e-07, 3.675113815630521e-07, 1.12282156642633e-07, -8.496579018571325e-07, -6.85980236555613e-07, 4.287740138376476e-07, 8.127381405661958e-08, -8.150221029162808e-07, -6.181120879341037e-07, 2.8923570699622293e-07, 1.1067748687239241e-08, -8.865671794546604e-07, -5.755911558260984e-07, 2.9651331390629047e-07, -4.8278924867451405e-08, -8.036760148099667e-07, -5.8760959768345e-07, 3.6302185324887546e-07, 4.103863964652152e-08, -8.744700195307281e-07, -5.922756674366135e-07, 2.891778689571345e-07, 3.371457579991281e-08, -8.654286468933301e-07, -6.191225868628107e-07, 2.7396004492208054e-07, -1.3941569237719591e-08, -8.497808005478407e-07, -6.558935629772889e-07, 2.6395460652176394e-07, -7.847126865975674e-09, -8.489671647372295e-07, -6.511114724333639e-07, 2.8341807910557e-07, -2.9233227410391212e-08, -8.5938729234993e-07, -6.785326183082284e-07, 4.354848280281538e-07, 2.7713160803848516e-07, -7.399486705884357e-07, -6.683484198656292e-07, 3.119723421529935e-07, 5.914182656089026e-08, -7.13435659308631e-07, -4.3841379322855666e-07, 4.972981071260369e-08, -2.0793514508368724e-07, -3.0190958661134345e-07, -4.0286177138474726e-07, 1.5776993757320848e-07, -2.469325569538296e-07, 3.440911961934084e-09, -6.468865288270066e-07], [-1.0785332103692183e-07, -3.488691461734495e-08, -2.474027509514669e-07, -5.415253025529337e-07, -1.431184948647965e-08, 3.57011869788617e-08, -5.12988335357809e-07, -3.516553209568175e-07, 4.884337784754892e-08, -5.320503187256069e-07, -5.652481687651946e-07, -2.728844759320664e-07, 4.6160042790707966e-07, -1.564820850783782e-07, -5.153293999537712e-07, -4.582123606523163e-07, 1.8447593705454596e-07, -1.8611479517224176e-07, -5.990680587561038e-07, -5.469487995384571e-07, 1.730082707017793e-07, -1.987033904696098e-07, -7.441941700741059e-07, -5.693812730389906e-07, 1.7838810843426799e-07, -1.6888127851565864e-07, -7.854250375410415e-07, -5.869352519442282e-07, 2.1387290344243754e-07, -1.2800942532138158e-07, -6.488381321042972e-07, -6.377545897355825e-07, 3.087451632486854e-07, -7.012805402721536e-08, -7.207505821731094e-07, -5.960646324079081e-07, 3.008341351626531e-07, -5.5762339253562045e-08, -7.122047725730148e-07, -5.916677113338691e-07, 3.4432010861381833e-07, -7.86648242206861e-08, -6.546892394235615e-07, -5.566030374073113e-07, 3.8143766813228867e-07, 2.9533361704307998e-08, -6.366320262103745e-07, -6.293198985305608e-07, 3.365027994374583e-07, 4.269620765151895e-10, -6.798585532247563e-07, -6.774026362292866e-07, 3.8861103501256974e-07, 7.384857596039366e-08, -5.987754721108033e-07, -7.116962769871626e-07, 4.043179504099276e-07, 1.6407845418411442e-07, -5.473235769132993e-07, -5.641786944286258e-07, 5.450473971835306e-07, 3.5917222417645647e-07, -5.81799427120385e-07, -6.24138082274132e-07, 3.7920789314981233e-07, 2.2755013422904904e-07, -5.80398485901323e-07, -7.255261236226136e-07, 4.543753051904828e-07, 9.816764100033596e-08, -6.052258414305744e-07, -6.113299655479683e-07, 4.008978549970519e-07, 1.6741008031801822e-07, -6.639983622174005e-07, -5.089075673112517e-07, 3.8211045063259207e-07, 9.867084106723057e-08, -7.731913901719097e-07, -6.207699181916601e-07, 4.374049386411515e-07, 7.342056657859008e-08, -7.423612712551096e-07, -5.622204138521405e-07, 3.1628056033187136e-07, 1.2710007338437856e-08, -8.02164707292926e-07, -5.264005691053015e-07, 3.202003906251629e-07, -4.161495898480059e-08, -7.32245930333279e-07, -5.379834041652226e-07, 3.789438735453102e-07, 3.750683417277984e-08, -7.955130702660217e-07, -5.420277197586475e-07, 3.141898363151513e-07, 3.214664104859582e-08, -7.862976749536168e-07, -5.652823777218778e-07, 3.0317469461927057e-07, -8.649681955122096e-09, -7.717076496196813e-07, -5.949454681832045e-07, 2.9369726689936494e-07, -6.182892302734946e-09, -7.751827414815361e-07, -5.909115547278933e-07, 3.121225159373856e-07, -2.475389910347358e-08, -7.827956768790006e-07, -6.175982865840175e-07, 4.4550195074865334e-07, 2.5904760344835875e-07, -6.717068568162241e-07, -6.091202334160738e-07, 3.315960849651825e-07, 4.433723945526465e-08, -6.67420521976306e-07, -4.25157601990237e-07, 8.605623008419402e-08, -2.1394961976847146e-07, -2.6500277100028846e-07, -3.760391631840622e-07, 2.2589846872363812e-07, -2.4175153034964625e-07, 6.322366414981049e-08, -6.29808764763954e-07], [-1.1574700131241798e-07, -4.2280132980302115e-08, -2.450858553609756e-07, -5.495311206783252e-07, -9.930595667394979e-10, 3.190767723585276e-08, -5.097812210071622e-07, -3.4569486187771695e-07, 7.041496856753577e-08, -5.033304634289377e-07, -5.453316029167643e-07, -2.466463964153683e-07, 4.76467123412051e-07, -1.4231813421322204e-07, -4.756240217120203e-07, -4.1897273968525467e-07, 2.190466672841054e-07, -1.6398897896661513e-07, -5.53199745372506e-07, -4.992671692958497e-07, 2.081449364811571e-07, -1.7519335585949134e-07, -6.85676900309889e-07, -5.194611612466364e-07, 2.1350544085020497e-07, -1.487025208823802e-07, -7.209341313666723e-07, -5.350712202779418e-07, 2.4458631239045705e-07, -1.1333962850231851e-07, -6.014828673717458e-07, -5.798928019106921e-07, 3.2863076717567625e-07, -6.210118561473113e-08, -6.641902633475548e-07, -5.433615605974405e-07, 3.222115660757731e-07, -4.958401538979075e-08, -6.563013273914364e-07, -5.393120738760498e-07, 3.600786243648169e-07, -7.068475040381727e-08, -6.056749565592469e-07, -5.094412467309855e-07, 3.9181680119010126e-07, 2.570910614137707e-08, -5.90410927480656e-07, -5.738236017557047e-07, 3.5271269174016127e-07, 7.737138497827859e-10, -6.270692018560446e-07, -6.164209196115815e-07, 3.988557219440434e-07, 6.673515358903232e-08, -5.558254810508188e-07, -6.45845513961892e-07, 4.1317074583292253e-07, 1.4647932307171304e-07, -5.109206600620791e-07, -5.166579447936444e-07, 5.354151696388874e-07, 3.1824997680611554e-07, -5.425321281165912e-07, -5.695764210493238e-07, 3.8991066777510483e-07, 2.0259763062908171e-07, -5.40324747671513e-07, -6.592003858878559e-07, 4.5608577787465833e-07, 8.793135737729203e-08, -5.636619812701409e-07, -5.568801115719312e-07, 4.09566206822785e-07, 1.4983277068604343e-07, -6.143122501651828e-07, -4.672881247335321e-07, 3.918409552277533e-07, 8.765649859297181e-08, -7.12051437932433e-07, -5.655778167908112e-07, 4.4154092073062015e-07, 6.615117824546673e-08, -6.85636423936414e-07, -5.142832984568421e-07, 3.3464897821856403e-07, 1.3332848212448604e-08, -7.36723029916997e-07, -4.827259529152452e-07, 3.378115285526419e-07, -3.5530196952192694e-08, -6.753585644566721e-07, -4.932822334975705e-07, 3.903420440999442e-07, 3.4639164080322165e-08, -7.312925289801268e-07, -4.97063653562292e-07, 3.331086932166357e-07, 2.9958420197488535e-08, -7.226168593767991e-07, -5.172575686984082e-07, 3.241192397035331e-07, -5.217690886060617e-09, -7.08965489787552e-07, -5.428729663712916e-07, 3.154788989857601e-07, -5.195427413629152e-09, -7.145573521922688e-07, -5.391402368246804e-07, 3.325250192771261e-07, -2.1498287527917113e-08, -7.203035019067979e-07, -5.641437879147662e-07, 4.516153260379594e-07, 2.4309853102926225e-07, -6.155030674060862e-07, -5.563293931217176e-07, 3.4578358555239793e-07, 3.45898705621601e-08, -6.291479937273023e-07, -4.1152805999018737e-07, 1.1894218220095161e-07, -2.154473902202852e-07, -2.3329092735516766e-07, -3.525005268212478e-07, 2.8178999469226204e-07, -2.3612978334330139e-07, 1.125889759201318e-07, -6.149379790597183e-07]]
bvp_hr = [[1.6276445493898808e-06, 1.6684745741001232e-06, 1.3668088924027964e-06, 1.0677559110683873e-06, 3.892064765790705e-07, 9.710067707782687e-09, 4.671076520196653e-08, 1.6868225919571097e-07, 4.171524977241979e-07, 6.240947336685932e-07, 5.365112504445431e-07, 4.489277892822479e-07, 3.6134456039228785e-07, 4.6481554574375045e-07, 3.1528896555959715e-07, -1.0512554739125837e-06, -3.101267468885135e-06, -5.3102517634914395e-06, -6.525746033103269e-06, -5.357089158963401e-06, -2.3952469758223582e-06, 1.1511189758427887e-06, 3.707824259380412e-06, 4.354291174561337e-06, 3.659156245346378e-06, 2.5845248351460717e-06, 1.1073692780556576e-06, 5.137905838866445e-07, 1.962372572258208e-07, 5.047359248342793e-07, 1.0477317656723655e-06, 1.2962021022303449e-06, 1.6711711579272254e-06, 1.6915860832805742e-06, 1.4590037009664303e-06, 7.914235944957429e-07, -1.065619430432166e-06, -3.5211590846483805e-06, -5.794699159864728e-06, -6.397723452202521e-06, -6.022754326785337e-06, -5.401229810460053e-06, -2.9230532747977997e-06, 1.6076035681241094e-07, 3.4356285296794697e-06, 4.211206871708965e-06, 4.204764818912025e-06, 3.0636629192051334e-06, 2.0260321100811864e-06, 1.1998709821452832e-06, 8.36262216999469e-07, 8.566772578970732e-07, 1.3811729134684942e-06, 1.7976700382073866e-06, 1.990639129185567e-06, 1.7255832924079736e-06, 7.1703139997295e-07, -1.780036671298253e-06, -4.474601174659599e-06, -6.052615246799187e-06, -6.333168036122126e-06, -5.9442259283700185e-06, -4.899761815587504e-06, -3.892298746928589e-06, -2.9698065520895013e-06, -1.8997024481319031e-06, -8.08769865400779e-08, 1.7794765994022157e-06, 3.3887471437965293e-06, 3.7618014874932765e-06, 3.2531937544695295e-06, 2.811056150608551e-06, 2.345891187263156e-06, 2.2167795921985875e-06, 2.4097488914108055e-06, 2.234191071794047e-06, 1.2716946179788784e-06, -7.443204689199749e-07, -2.8608031074115133e-06, -4.484872639236917e-06, -4.769952498492395e-06, -4.146513277085789e-06, -4.089097514460772e-06, -3.268161534476874e-06, -2.387197347183397e-06, -2.0122281028447016e-06, -2.251252774927092e-06, -2.7432749557339726e-06, -2.9407715784988955e-06, -2.735744166883703e-06, -9.335042240042215e-07, 1.835368559910172e-06, 4.562713361821632e-06, 5.632817490542968e-06, 5.653232633959258e-06, 5.084596362229989e-06, 3.972964353312182e-06, 1.5933403369128703e-06, -1.071754383663308e-06, -3.232768625780368e-06, -3.7047666794852495e-06, -3.288269653735456e-06, -2.3703046027037988e-06, -1.475367407356757e-06, -9.119564283521643e-07, -8.730410360029938e-07, -1.4851200814625125e-06, -2.1451690575444763e-06, -2.6787192978478028e-06, -2.7912451984004927e-06, -2.963799314091096e-06, -2.7153292291858663e-06, -2.4668587810312237e-06, -1.9238626762503313e-06, -7.272600530701323e-07, 4.028723969411477e-07, 2.437694790613335e-06, 3.0395023677341773e-06, 1.9591268948140257e-06, -1.66410505471764e-07, -1.660542605164103e-06, -2.235186106080563e-06, -2.2438057674596804e-06, -1.5383993415627166e-06, -1.2189313052933413e-06, -4.0443742438135934e-07, -2.2696490696048875e-07, -8.870139451360837e-07], [1.4596255751011079e-06, 1.4915128645623724e-06, 1.2178052773211417e-06, 9.511817515637304e-07, 3.5370596124082366e-07, 2.285373575041007e-08, 4.811107286851203e-08, 1.479961550491867e-07, 3.5816526212121095e-07, 5.326780400046692e-07, 4.5096624620779175e-07, 3.6925447166728e-07, 2.8754289724549944e-07, 3.7479916868007804e-07, 2.4148733060288415e-07, -9.452947971317676e-07, -2.7109841427526767e-06, -4.601816000407759e-06, -5.623003967765533e-06, -4.5899365857110434e-06, -2.0183777557886536e-06, 1.055230912642447e-06, 3.265825677085486e-06, 3.819475609621867e-06, 3.2126862925556972e-06, 2.2750446886432077e-06, 9.835231268117733e-07, 4.6067503054530307e-07, 1.814228374275392e-07, 4.398770083522089e-07, 9.062705872795199e-07, 1.1164397795244513e-06, 1.436892973264639e-06, 1.4528365454334503e-06, 1.2482122257713372e-06, 6.748494395449074e-07, -9.185569980676567e-07, -3.016242799314145e-06, -4.96575875042483e-06, -5.480437415975978e-06, -5.159984154483597e-06, -4.6326769396753606e-06, -2.4829958738067306e-06, 1.8844799496593477e-07, 3.0288599992947643e-06, 3.69987800374482e-06, 3.6861639651751444e-06, 2.686523249389907e-06, 1.7741388969228963e-06, 1.0466664374134586e-06, 7.213588125685382e-07, 7.373024836333205e-07, 1.191067535085663e-06, 1.547177115856247e-06, 1.7194601284489478e-06, 1.499977405253763e-06, 6.533023392151914e-07, -1.4800401454947675e-06, -3.7960646723548587e-06, -5.167758632415125e-06, -5.421753635441117e-06, -5.099070747194008e-06, -4.2019395453296925e-06, -3.3300659771831754e-06, -2.532820218530809e-06, -1.6055776089716652e-06, -3.69746075848928e-08, 1.5672848215208945e-06, 2.9542909990194596e-06, 3.271429171036703e-06, 2.8232089588175173e-06, 2.4369875576458505e-06, 2.027738741874502e-06, 1.910370523032793e-06, 2.0826537163333297e-06, 1.948197650749843e-06, 1.1475780194732604e-06, -5.709707277639929e-07, -2.39865931567083e-06, -3.816408732515379e-06, -4.080802524910325e-06, -3.5501802775790557e-06, -3.5089792499553848e-06, -2.7956747013856132e-06, -2.034085268181028e-06, -1.7136319131753295e-06, -1.9319704790512176e-06, -2.370877106754569e-06, -2.553559306888349e-06, -2.3823616477005654e-06, -8.230721966217743e-07, 1.5762033545307806e-06, 3.939822596012746e-06, 4.867065220924108e-06, 4.883008984658093e-06, 4.386503624827598e-06, 3.4236047636882695e-06, 1.3797482548701693e-06, -8.995344984083466e-07, -2.740995990191819e-06, -3.1349914753950203e-06, -2.7788819804115657e-06, -1.992035074883223e-06, -1.2282159992095763e-06, -7.458786645234321e-07, -7.173063230408408e-07, -1.2527830867671223e-06, -1.837630111354615e-06, -2.312193122498719e-06, -2.4202476415156386e-06, -2.5765870426814357e-06, -2.366418065208586e-06, -2.1562487779447827e-06, -1.6898549689575867e-06, -6.523285270034002e-07, 3.231990770536814e-07, 2.0871129082145176e-06, 2.6022456097952348e-06, 1.6546589992114648e-06, -1.9382770090597189e-07, -1.4656954233282873e-06, -1.9184339654974197e-06, -1.8817509689890397e-06, -1.2546175555292218e-06, -9.720503416435745e-07, -2.724598704604122e-07, -1.3468880244121795e-07, -7.195358836229736e-07], [1.3165918422489774e-06, 1.3312139215191957e-06, 1.0769404179269643e-06, 8.383835608281318e-07, 3.1107445430362214e-07, 2.2322234752859845e-08, 4.2178835535692693e-08, 1.253302114262535e-07, 3.0473228290637504e-07, 4.511783882142401e-07, 3.721669712358552e-07, 2.9315557171847176e-07, 2.1414434013456342e-07, 2.873672860527525e-07, 1.6808879255732873e-07, -8.702588781064884e-07, -2.4065456423433625e-06, -4.0458404638405136e-06, -4.925057684867038e-06, -4.006957350406576e-06, -1.739837055908437e-06, 9.65098653401332e-07, 2.9047223245904154e-06, 3.3828048583832465e-06, 2.842950323896171e-06, 2.0143435473993566e-06, 8.739568059663294e-07, 4.0994242213932514e-07, 1.6145726528278158e-07, 3.837436592349196e-07, 7.886031699593842e-07, 9.680053160286017e-07, 1.2436581337089258e-06, 1.250969123001671e-06, 1.0657788521584176e-06, 5.620512534106282e-07, -8.246185091507626e-07, -2.6459324851402563e-06, -4.341211034443384e-06, -4.788954898556604e-06, -4.5133019829384204e-06, -4.0556299685078336e-06, -2.168818880873199e-06, 1.8145274933632612e-07, 2.6839588189361336e-06, 3.2740086931704637e-06, 3.263526538007254e-06, 2.381553239569265e-06, 1.572802968196708e-06, 9.235982843294209e-07, 6.290576966412369e-07, 6.363687660787019e-07, 1.0313000948345815e-06, 1.3399089320621187e-06, 1.4895260250949189e-06, 1.2975784643413222e-06, 5.648074392457574e-07, -1.2894628106724322e-06, -3.3064495589087162e-06, -4.504342748924953e-06, -4.732971443128477e-06, -4.460489621830929e-06, -3.6837262790512253e-06, -2.926819763892514e-06, -2.233208065500925e-06, -1.417700650567568e-06, -2.9936504312602505e-08, 1.39078368320019e-06, 2.6163850882446894e-06, 2.8946551181239e-06, 2.493935622944292e-06, 2.1465824017873814e-06, 1.7762016928012625e-06, 1.6642342261335876e-06, 1.8138514749282126e-06, 1.6982979727357004e-06, 1.011582382671563e-06, -4.780952660137775e-07, -2.067748639170414e-06, -3.3116972689032758e-06, -3.55342516379961e-06, -3.0983703029452107e-06, -3.0712026257238972e-06, -2.4534310649685175e-06, -1.7927753219367254e-06, -1.5171223692061903e-06, -1.7127949863079381e-06, -2.1009689976259535e-06, -2.2636856299244555e-06, -2.1146224013215565e-06, -7.394036903721421e-07, 1.3780992917277593e-06, 3.462646318795101e-06, 4.278153737510045e-06, 4.285464901675244e-06, 3.8418610367049255e-06, 2.99339769284655e-06, 1.2091793547308015e-06, -7.742974807519187e-07, -2.370154165158387e-06, -2.708547813118311e-06, -2.399939055129804e-06, -1.7194266102901397e-06, -1.0619419695479971e-06, -6.497713209143919e-07, -6.325319539162513e-07, -1.1064746289382035e-06, -1.6238553405925445e-06, -2.044985384747732e-06, -2.1444072936696255e-06, -2.2867133771004774e-06, -2.1073114092261196e-06, -1.9279091806622058e-06, -1.5230494654361278e-06, -6.112915191404636e-07, 2.4710016944651137e-07, 1.8075092048155292e-06, 2.276204181774132e-06, 1.4554490678662283e-06, -1.6040315060134196e-07, -1.2708660803603282e-06, -1.660964676575976e-06, -1.6231758397931831e-06, -1.0780735676256774e-06, -8.32205624004217e-07, -2.2491627464932118e-07, -1.1198001596478599e-07, -6.29360786679567e-07]]
from scipy.signal import welch
def welch_method(signal, fps):
    frequencies, psd = welch(signal, fs=fps, nperseg=512, nfft=4096)

    first = np.where(frequencies > 0.7)[0]
    last = np.where(frequencies < 4)[0]

    first_index = first[0]
    last_index = last[-1]

    range_of_interest = range(first_index, last_index + 1, 1)
    max_idx = np.argmax(psd[range_of_interest])
    f_max = frequencies[range_of_interest[max_idx]]

    hr = f_max * 60.0

    return hr

# i = 2
#
# rPPG_hr = welch_method(rppg_hr[i], fps=30)
# bvp_hr = welch_method(bvp_hr[i], fps=30)
# print(rPPG_hr, bvp_hr)


def physnet_framework(input_video, ground_truth, training_length=128):
    """
    :param input_video:
    :param ground_truth:
    :param training_length:
    :return:
    """

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device} device")

    model = PhysNet_padding_Encoder_Decoder_MAX(frames=training_length).to(device)
    loss_function = Neg_Pearson().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    frames = np.array(extract_raw_sig(input_video, framework='PhysNet', width=1, height=1))

    video_data = frames[0:training_length]
    ground_truth = np.array(ground_truth[0:training_length])

    video_tensor = torch.from_numpy(video_data).permute(3, 0, 1, 2).float().unsqueeze(0)  # Add a batch dimension
    bvp_tensor = torch.from_numpy(ground_truth).float().unsqueeze(0)  # Add a batch dimension

    dataset = TensorDataset(video_tensor, bvp_tensor)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Start training
    model.train()
    num_epochs = 15
    for epoch in range(num_epochs):
        for video, bvp in data_loader:
            optimizer.zero_grad()

            video, bvp = video.to(device), bvp.to(device)

            rPPG, x_visual, x_visual3232, x_visual1616 = model(video)
            rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
            BVP_label = (bvp - torch.mean(bvp)) / torch.std(bvp)  # normalize

            loss = loss_function(rPPG, BVP_label)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


    # Define Mean Absolute Error
    def mae(prediction, target):
        return torch.mean(torch.abs(prediction - target))

    # Create test data loader
    # test_dataset = VideoDataset(test_video_folder, test_label_folder)
    # test_data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    test_dataset = TensorDataset(video_tensor, bvp_tensor)
    test_data_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Evaluation
    model.eval()  # Set the model to evaluation mode
    total_mae = 0
    with torch.no_grad():  # Do not calculate gradients for efficiency
        for video, bvp in test_data_loader:
            video = video.cuda()  # if using GPU
            bvp = bvp.cuda()  # if using GPU

            # Forward pass
            rPPG, x_visual, x_visual3232, x_visual1616 = model(video)

            # Calculate loss
            rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)
            bvp = (bvp - torch.mean(bvp)) / torch.std(bvp)
            total_mae += mae(rPPG, bvp).item()

    # Calculate average MAE
    avg_mae = total_mae / len(test_data_loader)
    print('The MAE of the test dataset is: ', avg_mae)
