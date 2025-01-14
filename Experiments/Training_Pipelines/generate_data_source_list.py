import os

f = open("./data_source_list.csv", "w")

def write(dataset, subfolder):

    if subfolder != "none":
        arr = os.listdir("./RPS_v1_data/"+dataset+"/Test/"+subfolder)
        for i in arr:
            f.write("/RPS_v1_data/"+dataset+"/Test/"+subfolder+"/"+i+","+subfolder+"\n")

        arr2 = os.listdir("./RPS_v1_data/" + dataset + "/Train/" + subfolder)
        for i in arr2:
            f.write("/RPS_v1_data/" + dataset + "/Train/"+subfolder +"/"+ i + "," + subfolder+"\n")

        arr3 = os.listdir("./RPS_v1_data/" + dataset + "/Validation/" + subfolder)
        for i in arr3:
            f.write("/RPS_v1_data/" + dataset + "/Validation/" + subfolder +"/"+ i + "," + subfolder+"\n")

def write_ngg(dataset, subfolder):
    arr = os.listdir("./RPS_v1_data/" + dataset +"/"+ subfolder)
    for i in arr:
        gesture = os.listdir("./RPS_v1_data/" + dataset +"/"+ subfolder+"/"+i)
        for j in gesture:
            f.write("/RPS_v1_data/" + dataset +"/"+ subfolder+"/"+i+"/"+j+",ngg"+"\n")

def write_ngg2(dataset, subfolder):
    arr = os.listdir("./RPS_v1_data/" + dataset +"/"+ subfolder)
    for i in arr:
        f.write("/RPS_v1_data/" + dataset +"/"+ subfolder+"/"+i+",ngg"+"\n")



dataset1 = os.listdir("RPS_v1_data/image_data/Test/")
for i in dataset1:
    write("image_data", i)

dataset2 = os.listdir("RPS_v1_data/rps_new_dataset/Test/")
for i in dataset2:
    write("rps_new_dataset", i)

dataset3 = os.listdir("RPS_v1_data/Dataset/Test/")
for i in dataset3:
    write("Dataset", i)

dataset4 = os.listdir("RPS_v1_data/Rock,Paper,ScissorsDataset/Test/")
for i in dataset4:
    write("Rock,Paper,ScissorsDataset", i)

dataset5 = os.listdir("RPS_v1_data/aug-rock-paper-scissors-dataset/Test/")
for i in dataset5:
    write("aug-rock-paper-scissors-dataset", i)




American_sign = os.listdir("RPS_v1_data/American Sign Language Digits Dataset")
for i in American_sign:
    if i != "readme.txt":
        write_ngg("American Sign Language Digits Dataset", i)

hagrid1 = os.listdir("RPS_v1_data/hagrid-classification-512p")
for i in hagrid1:
    if i != "readme.txt":
        write_ngg2("hagrid-classification-512p", i)

hagrid2 = os.listdir("RPS_v1_data/hagrid-sample-30k-384p")
for i in hagrid2:
    if i != "readme.txt":
        write_ngg("hagrid-sample-30k-384p", i)

hagrid3 = os.listdir("RPS_v1_data/hagrid-sample-120k-384p")
for i in hagrid3:
    if i != "readme.txt":
        write_ngg("hagrid-sample-120k-384p", i)

dataset_rgb = os.listdir("RPS_v1_data/Dataset_RGB")
for i in dataset_rgb:
    if i != "readme.txt":
        write_ngg("Dataset_RGB", i)

f.close()