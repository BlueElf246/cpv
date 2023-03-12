import os.path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import pickle
def save(file, filename):
    with open(f'{filename}.pkl','wb') as f:
        pickle.dump(file, f)
    print('save done')

# reference: https://github.com/praveenVnktsh/ViolaJones-EigenFace/blob/master/ViolaJonesDetectionWithEigenFaceRecognition.ipynb
def eigen_extraction(img_paths, k=8, id_list=None):
    face_matrix = []
    labels = []
    for id_face in id_list:
        p = img_paths+f"/id{id_face}/*.jpeg"
        img = glob.glob(p)
        for img_path in img:
            img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized= cv2.resize(img1, (322,322))
            i= np.array(cv2.equalizeHist(img_resized)).reshape(-1,)
            face_matrix.append(i)
            labels.append(id_face)
    face_matrix= np.array(face_matrix)
    print(face_matrix.shape)
    mean_face= face_matrix.mean(axis=0)
    mean_face_matrix= face_matrix - mean_face
    covMat = np.dot(mean_face_matrix, mean_face_matrix.T)
    eigenval, eigenvector= np.linalg.eig(covMat)
    idx= eigenval.argsort()[::-1]
    eigenval, eigenvector= eigenval[idx], eigenvector[idx]

    print(eigenvector.shape)
    eigenface= np.dot(eigenvector, mean_face_matrix)
    eigenface= eigenface[:k]
    print(eigenface.shape)

    W = np.zeros((face_matrix.shape[0], k))
    for i, mean_img in enumerate(mean_face_matrix):
        w = np.zeros((k,))
        for idx in range(len(w)):
            w[idx] = np.dot(mean_img.T, eigenface[idx].T)
        W[i] = w
    print(W.shape)
    params = {'database_id': W, 'mean_face': mean_face, 'eigenfaces': eigenface, 'label': labels, 'image_shape': (322, 322)}
    save(params, 'model')
def predict_image(img, training_dataset_encoded, mean_face, eigenFaces, label, image_shape):
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized= cv2.resize(img, image_shape)
    img_flat = np.array(cv2.equalizeHist(img_resized)).reshape(1,-1)
    img_mean= img_flat-mean_face
    encode_img= np.zeros((eigenFaces.shape[0],))
    for x in range(eigenFaces.shape[0]):
        encode_img[x]= np.dot(img_mean, eigenFaces[x])
    diff= np.linalg.norm(training_dataset_encoded- encode_img, axis=1)
    idx= diff.argmin()
    # print(diff)
    # print(diff[idx])
    return label[idx], diff[idx]

r=eigen_extraction("/Users/datle/Desktop/CPV/cpv/workshop8/eigenfaces/dataset", k=8, id_list=[1,2,3,4])
# img= cv2.imread("/Users/datle/Desktop/eigenfaces/dataset/id2/0.jpeg")
# params=pickle.load(open("model.pkl", 'rb'))
# database_encoded, mean_face, eigenFace, label, image_shape= params['database_id'], params['mean_face'], \
#                                                             params['eigenfaces'], params['label'], params['image_shape']

# print(predict_image(img,database_encoded, mean_face, eigenFace, label, image_shape))







