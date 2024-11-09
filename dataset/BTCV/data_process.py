import json
import os


# data_json = os.path.join(r'E:\SAM-Med2Dv2\dataset\BTCV\dataset.json')
# with open(data_json, 'r') as f:
#     dataset_dict = json.load(f)

# train_data = dataset_dict['training']
# test_data = dataset_dict['test']

# new_train = []
# for idx, data in enumerate(train_data):
#     if idx % 10 == 0:
#         new_train.append(
#             {
#                 'image': 'image/' + data['image'].split('imagesTr/')[1],
#                 'label': 'label/' + data['process_label'].split('prcess_labelsTr/')[1],
#                 'imask': 'imask/' + data['pseudo_label'].split('pseudo_labelsTr/')[1],
#             }
#         )

# new_test = []
# for idx, data in enumerate(test_data):
#     if idx % 10 == 0:
#         new_test.append(
#             {
#                 'image': 'image/' + data['image'].split('imagesTr/')[1],
#                 'label': 'label/' + data['process_label'].split('prcess_labelsTr/')[1],
#                 'imask': 'imask/' + data['pseudo_label'].split('pseudo_labelsTr/')[1],
#             }
#         )


# dataset_dict['numTraining'] = len(new_train)
# dataset_dict['training'] = new_train

# dataset_dict['numTest'] = len(new_test)
# dataset_dict['test'] = new_test

# with open(r'E:\SAM-Med2Dv2\dataset/BTCV/new_dataset.json', 'w') as file:
#     json.dump(dataset_dict, file, indent=4)


data_json = os.path.join(r'E:\SAM-Med2Dv2\dataset\BTCV\dataset.json')
with open(data_json, 'r') as f:
    dataset_dict = json.load(f)

train_image, train_label, train_imask = [], [], []
for data in dataset_dict['training']:
    train_image.append(data['image'].split('/')[-1])
    train_label.append(data['label'].split('/')[-1])
    train_imask.append(data['imask'].split('/')[-1])

test_image, test_label, test_imask = [], [], []
for data in dataset_dict['test']:
    test_image.append(data['image'].split('/')[-1])
    test_label.append(data['label'].split('/')[-1])
    test_imask.append(data['imask'].split('/')[-1])

all_image = os.listdir(r'E:\SAM-Med2Dv2\dataset\BTCV\image')
all_label = os.listdir(r'E:\SAM-Med2Dv2\dataset\BTCV\label')
all_imask = os.listdir(r'E:\SAM-Med2Dv2\dataset\BTCV\imask')

n = 0
for img in all_image:
    if img in train_image or img in test_image:
        continue
    else:
        os.remove(os.path.join(r'E:\SAM-Med2Dv2\dataset\BTCV\image', img))
    

for lab in all_label:
    if lab in train_label or img in test_label:
        continue
    else:
        os.remove(os.path.join(r'E:\SAM-Med2Dv2\dataset\BTCV\label', lab))

for imk in all_imask:
    if imk in train_imask or img in test_imask:
        continue
    else:
        os.remove(os.path.join(r'E:\SAM-Med2Dv2\dataset\BTCV\imask', imk))

