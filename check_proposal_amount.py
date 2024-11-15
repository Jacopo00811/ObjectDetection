import os


def count_files_in_folder(folder_path):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return len(files)


def check_proposal_amount(folder_path):
    result = {}
    for f in os.listdir(folder_path):
        if f.endswith('.xml'):
            pass
        else:
            result[f] = count_files_in_folder(os.path.join(folder_path, f, "positive"))

    count = 0
    for key, value in result.items():
        if value == 0:
            #print(key)
            count += 1
    # print(f"Number of files with no positive proposal: {count}")

    return count, result


script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, 'Potholes', 'annotated-images', 'test')
count, res = check_proposal_amount(folder_path)

# 532 images in train 
# 133 images in test



"""
PS C:\Users\akira\Desktop\ObjectDetection> python .\check_proposal_amount.py
img-12
img-129
img-145
img-153
img-154
img-160
img-174
img-191
img-194
img-208
img-213
img-224
img-235
img-246
img-254
img-261
img-262
img-27
img-281
img-36
img-401
img-423
img-438
img-467
img-481
img-51
img-546
img-620
img-629
img-632
img-64
img-91
img-92
img-96
Number of files with no positive proposal: 34

"""