import os
import ujson
def split_function(train_dataset,dev_dataset,test_dataset,dataset_files,cur_label):
    train_split_index = int(len(dataset_files) * 0.8)
    dev_split_index = train_split_index +  int(len(dataset_files)*0.1)
    cur_train = dataset_files[:train_split_index]
    cur_dev = dataset_files[train_split_index:dev_split_index]
    cur_test = dataset_files[dev_split_index:]
    for file in cur_train:
        train_dataset.append((file,cur_label))
    for file in cur_dev:
        dev_dataset.append((file,cur_label))
    for file in cur_test:
        test_dataset.append((file,cur_label))

    
def data_static_info():
    root_dir_path = "PiYingImage_v3cp/"
    chou_dir = root_dir_path + "1/"
    dan_dir = root_dir_path + "2/"
    jing_dir = root_dir_path + "3/"
    mo_dir = root_dir_path + "4/"
    sheng_dir = root_dir_path + "5/"
    
    chou_files = os.listdir(chou_dir)
    dan_files = os.listdir(dan_dir)
    jing_files = os.listdir(jing_dir)
    mo_files = os.listdir(mo_dir)
    sheng_files = os.listdir(sheng_dir)
    
    files_list = [chou_files,dan_files,jing_files,mo_files,sheng_files]

    train_dataset = []
    dev_dataset = []
    test_dataset = []
    
    for index,files in enumerate(files_list):
        split_function(train_dataset,dev_dataset,test_dataset,files,index+1)
    
    print(len(train_dataset))
    print(len(dev_dataset))
    print(len(test_dataset))
    ujson.dump(train_dataset,open("train_dataset.json",'w'),ensure_ascii=False)
    ujson.dump(dev_dataset,open("dev_dataset.json",'w'),ensure_ascii=False)
    ujson.dump(test_dataset,open("test_dataset.json",'w'),ensure_ascii=False)
    #看看是否改了


if __name__ == "__main__":
    data_static_info()
    

