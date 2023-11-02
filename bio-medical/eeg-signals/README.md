# EEG signals feature extraction and classification

1. Recoded electroencephalography (EGG) signals in 128 channels.
2. Features extracted using a sliding window of length 250 which slides 50 steps every time. The Fast Fourier Transform (FFT) was applied to the data inside each window on each channel indivitually, and then added together. 
3. A Logistic Regression (LR) model has been trained and evaluated using 5-fold method.