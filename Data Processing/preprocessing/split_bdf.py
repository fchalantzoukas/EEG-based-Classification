import mne
import pandas as pd
import os
from datetime import datetime

def createSubfolder(subfolder):
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

def loadEegData(bdfFile):
    raw = mne.io.read_raw_bdf(bdfFile, preload=True)
    usefulChannels = raw.ch_names[:34]
    raw = raw.pick_channels(usefulChannels)

    # Use average of mastoid channels as reference
    raw = raw.set_eeg_reference(ref_channels=['EXG1', 'EXG2'])
    usefulChannels = raw.ch_names[:32]
    raw = raw.pick_channels(usefulChannels)

    startDate = raw.info['meas_date']
    startTime = datetime.strptime(startDate.strftime("%H:%M:%S.%f")[:-3], "%H:%M:%S.%f")

    return raw, startTime

def splitAndSaveSegment(raw, timestampsDf, startTime, subfolder, userIndex):
    for index, row in timestampsDf.iterrows():
        startTimestamp = datetime.strptime(row['start'], "%H:%M:%S.%f")
        endTimestamp = datetime.strptime(row['end'], "%H:%M:%S.%f")

        startTimeFloat = (startTimestamp - startTime).total_seconds()
        endTimeFloat = (endTimestamp - startTime).total_seconds()

        segment = raw.copy().crop(tmin=startTimeFloat, tmax=endTimeFloat)

        outputFilePath = os.path.join(subfolder, f'user{userIndex}_segment_{index}_{row["username"]}.fif')
        segment.save(outputFilePath, overwrite=True)

def splitData(userIndex):
    subfolder = f"../user_extracted_data/user{userIndex}"
    createSubfolder(subfolder)

    csvName = f"../recordings/user{userIndex}.csv"
    columnNames = ['username', 'start', 'end']
    timestampsDf = pd.read_csv(csvName, names=columnNames)

    bdfFile = f'../recordings/user{userIndex}.bdf'
    raw, startTime = loadEegData(bdfFile)

    splitAndSaveSegment(raw, timestampsDf, startTime, subfolder, userIndex)
    raw.close()

if __name__ == "__main__":
    for i in range(1, 23):
        userIndex = f"0{i}" if i < 10 else str(i)
        splitData(userIndex)

