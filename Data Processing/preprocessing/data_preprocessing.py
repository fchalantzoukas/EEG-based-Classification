import mne
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def process_user_data(user_index):
    j = 29
    k = user_index + 1 if user_index not in [11, 22] else user_index - 10

    for i in range(30):
        if i < 20:
            data_file = f'../user_extracted_data/user{user_index:02}/user{user_index:02}_segment_{i}_user{user_index:02}.fif'
            numpy_file = f'../user_extracted_data/user{user_index:02}/user{user_index:02}_true_{i}.npy'
        else:
            data_file = f'../user_extracted_data/user{k:02}/user{k:02}_segment_{j}_user{user_index:02}.fif'
            numpy_file = f'../user_extracted_data/user{user_index:02}/user{user_index:02}_false_{i - 20}.npy'
            j -= 1
            k += 1
            if k > 11 and user_index < 12:
                k = 1
            elif k > 22:
                k = 12

        raw = load_and_preprocess_data(data_file)
        filtered_data = apply_moving_average(raw)
        X_scaled = scale_data(filtered_data)

        save_data(X_scaled, numpy_file)

    print(f"User {user_index:02} data ready")


def load_and_preprocess_data(data_file):
    raw = mne.io.read_raw_fif(data_file, preload=True, verbose='CRITICAL')
    freqs = (50, 100)

    raw.filter(l_freq=0.5, h_freq=128.0, method='iir', iir_params=dict(order=10, ftype='butter'), verbose='CRITICAL')
    raw.resample(sfreq=256.0)
    raw.notch_filter(freqs=freqs, method="spectrum_fit", verbose='CRITICAL')

    try:
        auto_noisy_chs, auto_flat_chs = mne.preprocessing.find_bad_channels_maxwell(raw, verbose='CRITICAL')
        raw.info['bads'] += auto_noisy_chs + auto_flat_chs
    except Exception as e:
        pass

    raw.set_eeg_reference(ref_channels='average', verbose='CRITICAL')

    return raw


def apply_moving_average(raw, window_size=6):
    eeg_data = raw.get_data()
    filtered_data = np.empty_like(eeg_data)

    for i in range(eeg_data.shape[0]):
        moving_avg = np.convolve(eeg_data[i, :], np.ones(window_size) / window_size, mode='same')
        filtered_data[i, :] = moving_avg

    return filtered_data


def scale_data(eeg_data):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(eeg_data)
    return X_scaled


def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        np.save(f, data)

if __name__ == '__main__':
    for user_index in range(1, 23):
        process_user_data(user_index)