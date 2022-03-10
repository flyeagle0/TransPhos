import csv
import numpy as np
import tensorflow.keras.utils as kutils

#input format   label,proteinName, postion,sites, shortsequence,
#input must be a .csv file
#positive_position_file_name is an csv file



def getMatrixInput(positive_position_file_name,sites, window_size=51, split = 0. ,empty_aa = '*'):
    # input format  proteinName, postion, shortsequence,
    prot = []  # list of protein name
    pos = []  # list of position with protein name
    rawseq = []
    label = []

    short_seqs = []
    half_len = (window_size - 1) / 2
    half_len = int(half_len)
    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:
            sseq = row[3]
            position = int(row[2])
            center = sseq[position-1]
            if center in sites:
                label.append(row[0])
                prot.append(row[1])
                pos.append(row[2])
            
                rawseq.append(sseq)        

                if position - half_len > 0:
                    start = position - half_len

                    left_seq = sseq[start - 1:position - 1]
                else:
                    left_seq = sseq[0:position - 1]

                end = len(sseq)
                if position + half_len < end:
                    end = position + half_len
                right_seq = sseq[position:end]

                if len(left_seq) < half_len:
                    nb_lack = half_len - len(left_seq)
                    left_seq = ''.join([empty_aa for count in range(nb_lack)]) + left_seq

                if len(right_seq) < half_len:
                    nb_lack = half_len - len(right_seq)
                    right_seq = right_seq + ''.join([empty_aa for count in range(nb_lack)])
                shortseq = left_seq + center + right_seq
                short_seqs.append(shortseq)

        all_label = [0] *5 + [1]*(len(short_seqs)-5)
        targetY = kutils.to_categorical(all_label)
        print(all_label)
        print(targetY)
        ONE_HOT_SIZE = 21
        letterDict = {}
        letterDict["A"] = 0
        letterDict["C"] = 1
        letterDict["D"] = 2
        letterDict["E"] = 3
        letterDict["F"] = 4
        letterDict["G"] = 5
        letterDict["H"] = 6
        letterDict["I"] = 7
        letterDict["K"] = 8
        letterDict["L"] = 9
        letterDict["M"] = 10
        letterDict["N"] = 11
        letterDict["P"] = 12
        letterDict["Q"] = 13
        letterDict["R"] = 14
        letterDict["S"] = 15
        letterDict["T"] = 16
        letterDict["V"] = 17
        letterDict["W"] = 18
        letterDict["Y"] = 19
        letterDict["*"] = 20

        Matr = np.zeros((len(short_seqs), window_size))
        samplenumber = 0
        for seq in short_seqs:
            AANo = 0
            for AA in seq:
                if AA not in letterDict:
                    continue
                Matr[samplenumber][AANo] = letterDict[AA]
                AANo = AANo+1
            samplenumber = samplenumber + 1
    return Matr, targetY, prot, pos, label
