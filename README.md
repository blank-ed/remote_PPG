# Remote Photoplethysmography
This is my implementations of rPPG algorithms. **_This is still a work in progress_**.

## Background on rPPG
RPPG stands for Remote Photoplethysmography, which is a technique that uses video cameras to measure physiological signals, such as the heart rate, from the face of a person. RPPG is a non-contact, non-invasive method that works by detecting tiny changes in skin color caused by the variation of blood volume in the skin vessels.

RPPG is based on the same principle as traditional photoplethysmography (PPG), which uses a sensor to measure the light absorption or reflection by the skin vessels. However, RPPG eliminates the need for physical contact with the skin, which makes it more comfortable and convenient for users. RPPG has applications in healthcare, wellness, and entertainment, such as monitoring vital signs, detecting stress, and enhancing user experience in video games and virtual reality.

The basic stages of rPPG are:
- **Video Acquisition**: A video camera is used to capture a video of the face of the subject. The video can be captured using different illumination sources, such as natural light or artificial light.
- **Region of interest (ROI) Selection**: A region of interest (ROI) is selected from the video, which typically includes the forehead, cheeks, or nose. The ROI should be large enough to capture the blood vessels but small enough to avoid the influence of surrounding tissues, such as wrinkles or movements near the eyes and the mouth.
- **Raw Signal Extraction**: The RGB color values of the pixels within the ROI are extracted from each frame of the video. The extracted signals are typically pre-processed to remove noise and artifacts, such as motion artifacts and variations in illumination. This can be done by using a lowpass or bandpass filter. Other methods of include detrending, normalizing or a combination of these methods. 
- **Pulse Signal Extraction**: Signal processing techniques, such as PCA, CHROM, spectral analysis, and machine learning, are used to analyze the extracted signals and extract the pulse signal from the raw signal.
- **Pulse Signal Filtering**: The pulse signal is typically bandpass filtered between 0.75 Hz (45 bpm) and 2 Hz (120 bpm) since the heart rate typically lies between this range.
- **Heart Rate Calculation**: The filtered pulse signal can be used to calculate the heart rate of the subject. This can be done using various methods, such as peak detection, zero-crossing detection, or spectral analysis.

The main purpose of this repository is to recreate the frameworks of all the state-of-the-art rPPG methods in order to make a meaningful comparison. Based on my research on this, I have found that there is an interdependency between the stages of rPPG. Therefore, in order to get a meaningful comparison with their frameworks, it is imperitive that every stage remains the same. 

Finished implementing ICA rPPG [link](https://opg.optica.org/oe/fulltext.cfm?uri=oe-18-10-10762&id=199381)

Currently working on PCA rPPG
