import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import Sequential, load_model, Model
from keras.metrics import categorical_accuracy
from Pokemon_ML_build import load_obj
from Image_processing import centralize_image, image_array_resize, rgb_hsv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


cwd = os.getcwd()
model_folder = os.path.join(cwd,"Models")
model_names = os.listdir(model_folder)

models = []
for each_model in model_names:
    folder_path = os.path.join("Models",each_model)
    full_m_path = os.path.join(cwd, folder_path)
    p_model = load_model(full_m_path)
    models.append(p_model)# last one is 100x100x3
    print(f"{each_model}: {p_model.layers[0].input}")



#DataSets
#Train Test
index_type_dict =  load_obj("Dictionary_index_to_type")
index_to_one_hot = load_obj('Dictionary_index_to_onehot')
database_dir = os.path.join(cwd,"Data/Train_test_Database")
file_name_X = "Database_6000_X_images.npy"
file_name_Y = "Database_6000_Y_labels.npy"
relative_path1 = os.path.join(database_dir,file_name_X)
relative_path2 = os.path.join(database_dir, file_name_Y)

# X = np.load(relative_path1, allow_pickle=True)
# Y = np.load(relative_path2, allow_pickle=True)
# print(X.shape, Y.shape)

# processed_X = np.array([centralize_image(j) for j in X]) #preprocess and center pics here

# p_X = np.array([image_array_resize(i, (64,64,3)) for i in processed_X]) #resize

# p_hsv_X = np.array([rgb_hsv(each_image) for each_image in p_X])

# #X_train, X_test, Y_train, Y_test = train_test_split(p_X,Y,test_size=0.3, random_state=1)
# #print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
# #Testing on training data
# # accuracies = []
# # for each_model in models[:len(models)-2]:
# #     pred = np.array(each_model.predict(X_train,batch_size=512))
# #     accuracy = np.mean(categorical_accuracy(Y_train, pred))
# #     accuracies.append(accuracy)
# # #HSV Model
# # pred_hsv = np.array(models[-2].predict(X_hsv_train, batch_size=512))
# # accuracies.append(np.mean(categorical_accuracy(Y_train, pred_hsv)))
# #print(accuracies)
# #No normalization model
# #   Resize
# # pred = np.array(poke_model.predict(X_test, batch_size=512))
# # pred_hsv = np.array(poke_model_hsv.predict(X_hsv_train, batch_size=512))
# # print(np.mean(categorical_accuracy(Y_test, pred)))
# # print(np.mean(categorical_accuracy(Y_train, pred_hsv)))
# pred = np.array(models[0].predict(np.array([X_train[0]]),verbose=1))
# print(pred)
# pred_mean = pred[0].mean()
# predo = np.argmax(pred_mean)
# print(predo)


# Confusion Matrixes
#cnf = confusion_matrix(Y_train, np.array(models[0].predict(X_train,batch_size=512)))
#print(cnf)
#Inner Layers

#Validate
v_database_dir = os.path.join(cwd,"Data/Train_test_Database")
v_file_name_X = "kaggle1_images_validation.npy"
v_file_name_Y = "kaggle1_labels_validation.npy"
v_relative_pathX = os.path.join(database_dir,file_name_X)
v_relative_pathY = os.path.join(database_dir, file_name_Y)

v_X = np.load(relative_path1, allow_pickle=True)
v_Y = np.load(relative_path2, allow_pickle=True)
print(v_X.shape, v_Y.shape)

v_processed_X = np.array([centralize_image(j) for j in v_X]) #preprocess and center pics here

v_p_X = np.array([image_array_resize(i, (64,64,3)) for i in v_processed_X]) #resize

v_p_hsv_X = np.array([rgb_hsv(each_image) for each_image in v_p_X])

# v_accuracies = []
# for each_model in models[:len(models)-2]:
#     v_pred = np.array(each_model.predict(v_p_X,batch_size=512))
#     v_accuracy = np.mean(categorical_accuracy(v_Y, v_pred))
#     v_accuracies.append(v_accuracy)

# print(v_accuracies)

hsv_model = models[-2]

predhsv = np.array(models[-2].predict(v_p_hsv_X, batch_size=512))
accuracyhsv = np.mean(categorical_accuracy(v_Y, predhsv))

v_X_100 = np.array([image_array_resize(i, (100,100,3)) for i in v_processed_X])
m100x100 = models[-1]
pred100 = np.array(m100x100.predict(v_X_100, batch_size=512))
aCCURACY100 = np.mean(categorical_accuracy(v_Y, pred100))

print(accuracyhsv, aCCURACY100)



