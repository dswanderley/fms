import csv


data_csv = '../dataset/train/labels.csv'  

labels = list()
sequences = [0]
count = 0
with open(data_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(reader):
            if i > 0:
                seq = int(row[0])
                frame = int(row[1])
                gt = 0 if len(row) < 3 else 1
                count += gt
                labels.append([seq, frame, gt])
                if seq > sequences[-1]:
                    sequences.append(seq)

data_ = []
c_count_c = []
c_count_l2 = []
c_count_l1 = []
c_count_r1 = []
c_count_r2 = []
b_count_c = []
b_count_l2 = []
b_count_l1 = []
b_count_r1 = []
b_count_r2 = []
c_count_11 = []
c_count_10 = []
c_count_00 = []
c_count_01 = []
for seq in sequences:
    seq_frames = [ [s,f,gt] for (s,f,gt) in labels if s == seq ]

    for i, data in enumerate(seq_frames):
        
        left_2 = 0
        left_1 = 0
        right_1 = 0
        right_2 = 0

        if i > 0:            
            left_1 = seq_frames[i-1][2]
        if i > 1:            
            left_2 = seq_frames[i-2][2]
        if i < len(seq_frames)-2:            
            right_2 = seq_frames[i+2][2]
        if i < len(seq_frames)-1:            
            right_1 = seq_frames[i+1][2]

        data_.append([data[0], data[1], left_2, left_1, data[2], right_1, right_2])

        if data[2] == 1:
            c_count_c.append([seq, i])
            if left_2 == 1:
                c_count_l2.append([seq, i])
            if left_1 == 1:
                c_count_l1.append([seq, i])
            if right_1 == 1:
                c_count_r1.append([seq, i])
            if right_2 == 1:
                c_count_r2.append([seq, i])

            if left_1 == 0 and right_1 == 0:
                c_count_00.append([seq, i])
            if left_1 == 0 and right_1 == 1:
                c_count_01.append([seq, i])
            if left_1 == 1 and right_1 == 0:
                c_count_10.append([seq, i])
            if left_1 == 1 and right_1 == 1:
                c_count_11.append([seq, i]) 


        if data[2] == 0:
            b_count_c.append([seq, i])
            if left_2 == 1:
                b_count_l2.append([seq, i])
            if left_1 == 1:
                b_count_l1.append([seq, i])
            if right_1 == 1:
                b_count_r1.append([seq, i])
            if right_2 == 1:
                b_count_r2.append([seq, i])

      


print('left  -2: ', len(c_count_l2)/len(c_count_c))
print('left  -1: ', len(c_count_l1)/len(c_count_c))
print('center 0: ', len(c_count_c)/len(c_count_c))
print('right +1: ', len(c_count_r1)/len(c_count_c))
print('right +2: ', len(c_count_r2)/len(c_count_c))

print('')

print('left  -2: ', len(b_count_l2)/len(b_count_c))
print('left  -1: ', len(b_count_l1)/len(b_count_c))
print('center 0: ', len(b_count_c)/len(b_count_c))
print('right +1: ', len(b_count_r1)/len(b_count_c))
print('right +2: ', len(b_count_r2)/len(b_count_c))

print('')



print('0 1 0: ', len(c_count_00)/len(c_count_c))
print('0 1 1: ', len(c_count_01)/len(c_count_c))
print('1 1 0: ', len(c_count_10)/len(c_count_c))
print('1 1 1: ', len(c_count_11)/len(c_count_c))

