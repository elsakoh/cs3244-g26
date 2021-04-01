from numpy.random import seed
seed(1)
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import h5py
import scipy.io as sio
import cv2
import glob
import gc
import tensorflow as tf

#from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from models import VGG16, lstm, mlp

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# CHANGE THESE VARIABLES ---
data_folder = '/home/abhinav/Desktop/ML/cs3244-g26/datasets/URFD_optical_flow'
mean_file = 'flow_mean.mat'
vgg_16_weights = 'weights.h5'

L = 10

# Set to 'True' if you want to restore a previous trained models
# Training is skipped and test is done
use_checkpoint = False # Set to True or False
# --------------------------

best_model_path = 'models/'
plots_folder = 'plots/'
checkpoint_path = best_model_path + 'fold_'

saved_files_folder = 'saved_features/'
features_file = saved_files_folder + 'features_urfd_tf.h5'
labels_file = saved_files_folder + 'labels_urfd_tf.h5'
features_key = 'features'
labels_key = 'labels'

# test and activate GPU
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print(tf.__version__)                         # -> 2.1.0
print(tf.config.list_physical_devices('GPU')) # -> [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
print(tf.test.is_built_with_cuda())           # -> True

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def sample_data(data, sample_size):
    # Where b is the ndarray
    # Get number of columns
    number_of_rows = data[0].shape[1]
    # Generate random column indexes
    random_indices = np.random.choice(number_of_rows, size=sample_size, replace=False)
    # Index the random cols
    randomised_data = []
    for d in data:
        randomised_data.append(d[:, random_indices])
    return randomised_data

def plot_training_info(case, metrics, save, history):
    '''
    Function to create plots for train and validation loss and accuracy
    Input:
    * case: name for the plot, an 'accuracy.png' or 'loss.png'
	will be concatenated after the name.
    * metrics: list of metrics to store: 'loss' and/or 'accuracy'
    * save: boolean to store the plots or only show them.
    * history: History object returned by the Keras fit function.
    '''
    val = False
    if 'val_acc' in history and 'val_loss' in history:
        val = True
    plt.ioff()
    if 'accuracy' in metrics:
        fig = plt.figure()
        plt.plot(history['accuracy'])
        if val: plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        if val:
            plt.legend(['train', 'val'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
        if save == True:
            plt.savefig(case + 'accuracy.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)

    # summarize history for loss
    if 'loss' in metrics:
        fig = plt.figure()
        plt.plot(history['loss'])
        if val: plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        if val:
            plt.legend(['train', 'val'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
        if save == True:
            plt.savefig(case + 'loss.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)

def generator(list1, lits2):
    '''
    Auxiliar generator: returns the ith element of both given list with
	 each call to next()
    '''
    for x,y in zip(list1,lits2):
        yield x, y

def generatorSimple(list):
    '''
    Auxiliar generator: returns the ith element of both given list with
	 each call to next()
    '''
    for x in zip(list):
        yield x

def saveFeaturesOtherDatasets(feature_extractor,
		 features_file,
		 labels_file,
		 features_key,
		 labels_key,
        num_features):
    '''
    Function to load the optical flow stacks, do a feed-forward through the
	 feature extractor (VGG16) and
    store the output feature vectors in the file 'features_file' and the
	labels in 'labels_file'.
    Input:
    * feature_extractor: model VGG16 until the fc6 layer.
    * features_file: path to the hdf5 file where the extracted features are
	 going to be stored
    * labels_file: path to the hdf5 file where the labels of the features
	 are going to be stored
    * features_key: name of the key for the hdf5 file to store the features
    * labels_key: name of the key for the hdf5 file to store the labels
    '''

    class0 = 'Falls'
    class1 = 'NotFalls'

    # Load the mean file to subtract to the images
    d = sio.loadmat(mean_file)
    flow_mean = d['image_mean']

    # Fill the folders and classes arrays with all the paths to the data
    folders, classes = [], []
    fall_videos = [f for f in os.listdir(data_folder + class0)
			if os.path.isdir(os.path.join(data_folder + class0, f))]
    fall_videos.sort()
    for fall_video in fall_videos:
        images = glob.glob(data_folder + class0 + '/' + fall_video
				 + '/flow_*.jpg')
        if int(len(images)) >= 10:
            folders.append(data_folder + class0 + '/' + fall_video)
            classes.append(0)

    not_fall_videos = [f for f in os.listdir(data_folder + class1)
			if os.path.isdir(os.path.join(data_folder + class1, f))]
    not_fall_videos.sort()
    for not_fall_video in not_fall_videos:
        images = glob.glob(data_folder + class1 + '/' + not_fall_video
				 + '/flow_*.jpg')
        if int(len(images)) >= 10:
            folders.append(data_folder + class1 + '/' + not_fall_video)
            classes.append(1)

    # Total amount of stacks, with sliding window = num_images-L+1
    nb_total_stacks = 0
    for folder in folders:
        images = glob.glob(folder + '/flow_*.jpg')
        nb_total_stacks += len(images)-L+1

    # File to store the extracted features and datasets to store them
    # IMPORTANT NOTE: 'w' mode totally erases previous data
    h5features = h5py.File(features_file,'w')
    h5labels = h5py.File(labels_file,'w')
    dataset_features = h5features.create_dataset(features_key,
			 shape=(nb_total_stacks, num_features),
			 dtype='float64')
    dataset_labels = h5labels.create_dataset(labels_key,
			 shape=(nb_total_stacks, 1),
			 dtype='float64')
    cont = 0

    for folder, label in zip(folders, classes):
        images = glob.glob(folder + '/flow_*.jpg')
        images.sort()
        nb_stacks = len(images)-L+1
        # Here nb_stacks optical flow stacks will be stored
        flow = np.zeros(shape=(224,224,2*L,nb_stacks), dtype=np.float64)
        gen = generatorSimple(images)
        for i in range(len(images)):
            flow_file = gen.next()
            img = cv2.imread(flow_file, cv2.IMREAD_GRAYSCALE)
            # Assign an image i to the jth stack in the kth position, but also
	    # in the j+1th stack in the k+1th position and so on
	    # (for sliding window)
            for s in list(reversed(range(min(10,i+1)))):
                if i-s < nb_stacks:
                    flow[:,:,2*s,  i-s] = img
            del img
            gc.collect()

        # Subtract mean
        flow = flow - np.tile(flow_mean[...,np.newaxis],
			      (1, 1, 1, flow.shape[3]))
        flow = np.transpose(flow, (3, 0, 1, 2))
        predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)
        truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
        # Process each stack: do the feed-forward pass and store
	# in the hdf5 file the output
        for i in range(flow.shape[0]):
            prediction = feature_extractor.predict(
					np.expand_dims(flow[i, ...],0))
            predictions[i, ...] = prediction
            truth[i] = label
        dataset_features[cont:cont+flow.shape[0],:] = predictions
        dataset_labels[cont:cont+flow.shape[0],:] = truth
        cont += flow.shape[0]
    h5features.close()
    h5labels.close()

def saveFeatures(feature_extractor,
		 features_file,
		 labels_file,
		 features_key,
		 labels_key,
        num_features):
    '''
    Function to load the optical flow stacks, do a feed-forward through the
	 feature extractor (VGG16) and
    store the output feature vectors in the file 'features_file' and the
	labels in 'labels_file'.
    Input:
    * feature_extractor: model VGG16 until the fc6 layer.
    * features_file: path to the hdf5 file where the extracted features are
	 going to be stored
    * labels_file: path to the hdf5 file where the labels of the features
	 are going to be stored
    * features_key: name of the key for the hdf5 file to store the features
    * labels_key: name of the key for the hdf5 file to store the labels
    '''

    class0 = 'Falls'
    class1 = 'NotFalls'

    # Load the mean file to subtract to the images
    d = sio.loadmat(mean_file)
    flow_mean = d['image_mean']

    # Fill the folders and classes arrays with all the paths to the data
    folders, classes = [], []
    fall_videos = [f for f in os.listdir(data_folder + class0)
			if os.path.isdir(os.path.join(data_folder + class0, f))]
    fall_videos.sort()
    for fall_video in fall_videos:
        x_images = glob.glob(data_folder + class0 + '/' + fall_video
				 + '/flow_x*.jpg')
        if int(len(x_images)) >= 10:
            folders.append(data_folder + class0 + '/' + fall_video)
            classes.append(0)

    not_fall_videos = [f for f in os.listdir(data_folder + class1)
			if os.path.isdir(os.path.join(data_folder + class1, f))]
    not_fall_videos.sort()
    for not_fall_video in not_fall_videos:
        x_images = glob.glob(data_folder + class1 + '/' + not_fall_video
				 + '/flow_x*.jpg')
        if int(len(x_images)) >= 10:
            folders.append(data_folder + class1 + '/' + not_fall_video)
            classes.append(1)

    # Total amount of stacks, with sliding window = num_images-L+1
    nb_total_stacks = 0
    for folder in folders:
        x_images = glob.glob(folder + '/flow_x*.jpg')
        nb_total_stacks += len(x_images)-L+1

    # File to store the extracted features and datasets to store them
    # IMPORTANT NOTE: 'w' mode totally erases previous data
    h5features = h5py.File(features_file,'w')
    h5labels = h5py.File(labels_file,'w')
    dataset_features = h5features.create_dataset(features_key,
			 shape=(nb_total_stacks, num_features),
			 dtype='float64')
    dataset_labels = h5labels.create_dataset(labels_key,
			 shape=(nb_total_stacks, 1),
			 dtype='float64')
    cont = 0

    for folder, label in zip(folders, classes):
        x_images = glob.glob(folder + '/flow_x*.jpg')
        x_images.sort()
        y_images = glob.glob(folder + '/flow_y*.jpg')
        y_images.sort()
        nb_stacks = len(x_images)-L+1
        # Here nb_stacks optical flow stacks will be stored
        flow = np.zeros(shape=(224,224,2*L,nb_stacks), dtype=np.float64)
        gen = generator(x_images,y_images)
        for i in range(len(x_images)):
            flow_x_file, flow_y_file = gen.next()
            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
            # Assign an image i to the jth stack in the kth position, but also
	    # in the j+1th stack in the k+1th position and so on
	    # (for sliding window)
            for s in list(reversed(range(min(10,i+1)))):
                if i-s < nb_stacks:
                    flow[:,:,2*s,  i-s] = img_x
                    flow[:,:,2*s+1,i-s] = img_y
            del img_x,img_y
            gc.collect()

        # Subtract mean
        flow = flow - np.tile(flow_mean[...,np.newaxis],
			      (1, 1, 1, flow.shape[3]))
        flow = np.transpose(flow, (3, 0, 1, 2))
        predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)
        truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
        # Process each stack: do the feed-forward pass and store
	# in the hdf5 file the output
        for i in range(flow.shape[0]):
            prediction = feature_extractor.predict(
					np.expand_dims(flow[i, ...],0))
            predictions[i, ...] = prediction
            truth[i] = label
        dataset_features[cont:cont+flow.shape[0],:] = predictions
        dataset_labels[cont:cont+flow.shape[0],:] = truth
        cont += flow.shape[0]
    h5features.close()
    h5labels.close()

def test_video(feature_extractor, video_path, ground_truth):
    num_features=4096
    # Load the mean file to subtract to the images
    d = sio.loadmat(mean_file)
    flow_mean = d['image_mean']

    x_images = glob.glob(video_path + '/flow_x*.jpg')
    x_images.sort()
    y_images = glob.glob(video_path + '/flow_y*.jpg')
    y_images.sort()
    nb_stacks = len(x_images)-L+1
    # Here nb_stacks optical flow stacks will be stored
    flow = np.zeros(shape=(224,224,2*L,nb_stacks), dtype=np.float64)
    gen = generator(x_images,y_images)
    for i in range(len(x_images)):
        flow_x_file, flow_y_file = gen.next()
        img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
        img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
        # Assign an image i to the jth stack in the kth position, but also
	# in the j+1th stack in the k+1th position and so on
	# (for sliding window)
        for s in list(reversed(range(min(10,i+1)))):
            if i-s < nb_stacks:
                flow[:,:,2*s,  i-s] = img_x
                flow[:,:,2*s+1,i-s] = img_y
        del img_x,img_y
        gc.collect()
    flow = flow - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow.shape[3]))
    flow = np.transpose(flow, (3, 0, 1, 2))
    predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)
    truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
    # Process each stack: do the feed-forward pass
    for i in range(flow.shape[0]):
        prediction = feature_extractor.predict(np.expand_dims(flow[i, ...],0))
        predictions[i, ...] = prediction
        truth[i] = ground_truth
    return predictions, truth

def train(use_validation=False, use_val_for_training = False, num_features=4096,
          learning_rate=0.0001, epochs=3000, threshold=0.5, exp='', batch_norm=True,
          mini_batch_size=64, save_plots=True, save_features=False, classification_method='MLP',
          val_size=10, weight_0=1, dataset_type=0):
    model = VGG16(num_features)
    # ========================================================================
    # WEIGHT INITIALIZATION
    # ========================================================================
    layerscaffe = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
		   'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
		   'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
    h5 = h5py.File(vgg_16_weights, 'r')

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Copy the weights stored in the 'vgg_16_weights' file to the
    # feature extractor part of the VGG16
    for layer in layerscaffe[:-3]:
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (2,3,1,0))
        w2 = w2[::-1, ::-1, :, :]
        b2 = np.asarray(b2)
        layer_dict[layer].set_weights((w2, b2))

    # Copy the weights of the first fully-connected layer (fc6)
    layer = layerscaffe[-3]
    w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
    w2 = np.transpose(np.asarray(w2), (1,0))
    b2 = np.asarray(b2)
    layer_dict[layer].set_weights((w2, b2))

    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================
    if save_features:
        if dataset_type == 0:
            saveFeatures(model, features_file, labels_file, features_key, labels_key, num_features)
        else:
            saveFeaturesOtherDatasets(model, features_file, labels_file, features_key, labels_key, num_features)

    # ========================================================================
    # TRAINING
    # ========================================================================  

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
  
    h5features = h5py.File(features_file, 'r')
    h5labels = h5py.File(labels_file, 'r')
    
    # X_full will contain all the feature vectors extracted
    # from optical flow images
    X_full = h5features[features_key]
    _y_full = np.asarray(h5labels[labels_key])

    zeroes_full = np.asarray(np.where(_y_full==0)[0])
    ones_full = np.asarray(np.where(_y_full==1)[0])
    zeroes_full.sort()
    ones_full.sort()
    
    # Use a 5 fold cross-validation
    kf_falls = KFold(n_splits=5, shuffle=True)
    kf_falls.get_n_splits(X_full[zeroes_full, ...])
    
    kf_nofalls = KFold(n_splits=5, shuffle=True)
    kf_nofalls.get_n_splits(X_full[ones_full, ...])        

    sensitivities = []
    specificities = []
    fars = []
    mdrs = []
    accuracies = []
        
    fold_number = 1
    # CROSS-VALIDATION: Stratified partition of the dataset into
    # train/test sets
    for ((train_index_falls, test_index_falls),
    (train_index_nofalls, test_index_nofalls)) in zip(
        kf_falls.split(X_full[zeroes_full, ...]),
        kf_nofalls.split(X_full[ones_full, ...])
    ):

        train_index_falls = np.asarray(train_index_falls)
        test_index_falls = np.asarray(test_index_falls)
        train_index_nofalls = np.asarray(train_index_nofalls)
        test_index_nofalls = np.asarray(test_index_nofalls)

        X = np.concatenate((
            X_full[zeroes_full, ...][train_index_falls, ...],
            X_full[ones_full, ...][train_index_nofalls, ...]
        ))
        _y = np.concatenate((
            _y_full[zeroes_full, ...][train_index_falls, ...],
            _y_full[ones_full, ...][train_index_nofalls, ...]
        ))
        X_test = np.concatenate((
            X_full[zeroes_full, ...][test_index_falls, ...],
            X_full[ones_full, ...][test_index_nofalls, ...]
        ))
        y_test = np.concatenate((
            _y_full[zeroes_full, ...][test_index_falls, ...],
            _y_full[ones_full, ...][test_index_nofalls, ...]
        ))

        if use_validation:
            # Create a validation subset from the training set
            zeroes = np.asarray(np.where(_y==0)[0])
            ones = np.asarray(np.where(_y==1)[0])
            
            zeroes.sort()
            ones.sort()

            trainval_split_0 = StratifiedShuffleSplit(n_splits=1,
                            test_size=int(val_size/2),
                            random_state=7)
            indices_0 = trainval_split_0.split(X[zeroes,...],
                            np.argmax(_y[zeroes,...], 1))
            trainval_split_1 = StratifiedShuffleSplit(n_splits=1,
                            test_size=int(val_size/2),
                            random_state=7)
            indices_1 = trainval_split_1.split(X[ones,...],
                            np.argmax(_y[ones,...], 1))
            train_indices_0, val_indices_0 = indices_0.__next__()
            train_indices_1, val_indices_1 = indices_1.__next__()

            X_train = np.concatenate([X[zeroes,...][train_indices_0,...],
                        X[ones,...][train_indices_1,...]],axis=0)
            y_train = np.concatenate([_y[zeroes,...][train_indices_0,...],
                        _y[ones,...][train_indices_1,...]],axis=0)
            X_val = np.concatenate([X[zeroes,...][val_indices_0,...],
                        X[ones,...][val_indices_1,...]],axis=0)
            y_val = np.concatenate([_y[zeroes,...][val_indices_0,...],
                        _y[ones,...][val_indices_1,...]],axis=0)
        else:
            X_train = X
            y_train = _y

        # Balance the number of positive and negative samples so that
        # there is the same amount of each of them
        all0 = np.asarray(np.where(y_train==0)[0])
        all1 = np.asarray(np.where(y_train==1)[0])  

        if len(all0) < len(all1):
            all1 = np.random.choice(all1, len(all0), replace=False)
        else:
            all0 = np.random.choice(all0, len(all1), replace=False)
        allin = np.concatenate((all0.flatten(),all1.flatten()))
        allin.sort()
        X_train = X_train[allin,...]
        y_train = y_train[allin]
    
        # ==================== CLASSIFIER ========================
        if classification_method == 'MLP':
            classifier = mlp(num_features, batch_norm)
        else:
            new_feature_length = int(num_features / 4)
            data = sample_data([X_train, X_test, X_val], new_feature_length)
            X_train = data[0]
            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = data[1]
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
            X_val = data[2]
            X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
            classifier = lstm(seq_length=1, feature_length=new_feature_length, nb_classes=1)

        fold_best_model_path = best_model_path + 'urfd_fold_{}.h5'.format(
                                fold_number)
        classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        if not use_checkpoint:
            # ==================== TRAINING ========================     
            # weighting of each class: only the fall class gets
            # a different weight
            class_weight = {0: weight_0, 1: 1}

            callbacks = None
            if use_validation:
                # callback definition
                metric = 'val_loss'
                e = EarlyStopping(monitor=metric, min_delta=0, patience=2,
                        mode='auto')
                c = ModelCheckpoint(fold_best_model_path, monitor=metric,
                            save_best_only=True,
                            save_weights_only=False, mode='auto')
                callbacks = [e, c]
            validation_data = None
            if use_validation:
                validation_data = (X_val,y_val)
            _mini_batch_size = mini_batch_size
            if mini_batch_size == 0:
                _mini_batch_size = X_train.shape[0]

            history = classifier.fit(
                X_train, y_train, 
                validation_data=validation_data,
                batch_size=_mini_batch_size,
                epochs=epochs,
                shuffle=True,
                class_weight=class_weight,
                callbacks=callbacks
            )

 #           if not use_validation:
 #              classifier.save(fold_best_model_path)

            plot_training_info(plots_folder + exp, ['accuracy', 'loss'],
                    save_plots, history.history)

            if use_validation and use_val_for_training:
                #classifier = load_model(fold_best_model_path)

                # Use full training set (training+validation)
                X_train = np.concatenate((X_train, X_val), axis=0)
                y_train = np.concatenate((y_train, y_val), axis=0)

                history = classifier.fit(
                    X_train, y_train, 
                    validation_data=validation_data,
                    batch_size=_mini_batch_size,
                    epochs=epochs,
                    shuffle='batch',
                    class_weight=class_weight,
                    callbacks=callbacks
                )

                classifier.save(fold_best_model_path)

        # ==================== EVALUATION ========================     
        
        # Load best model
        #print('Model loaded from checkpoint')
        #classifier = load_model(fold_best_model_path)

        predicted = classifier.predict(np.asarray(X_test))
        for i in range(len(predicted)):
            if predicted[i] < threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        # Array of predictions 0/1
        predicted = np.asarray(predicted).astype(int)   
        # Compute metrics and print them
        cm = confusion_matrix(y_test, predicted,labels=[0,1])
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        tpr = tp/float(tp+fn)
        fpr = fp/float(fp+tn)
        fnr = fn/float(fn+tp)
        tnr = tn/float(tn+fp)
        precision = tp/float(tp+fp)
        recall = tp/float(tp+fn)
        specificity = tn/float(tn+fp)
        f1 = 2*float(precision*recall)/float(precision+recall)
        accuracy = accuracy_score(y_test, predicted)
        
        print('FOLD {} results:'.format(fold_number))
        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp,tn,fp,fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(
                        tpr,tnr,fpr,fnr))   
        print('Sensitivity/Recall: {}'.format(recall))
        print('Specificity: {}'.format(specificity))
        print('Precision: {}'.format(precision))
        print('F1-measure: {}'.format(f1))
        print('Accuracy: {}'.format(accuracy))
        
        # Store the metrics for this epoch
        sensitivities.append(tp/float(tp+fn))
        specificities.append(tn/float(tn+fp))
        fars.append(fpr)
        mdrs.append(fnr)
        accuracies.append(accuracy)
        fold_number += 1

    print('5-FOLD CROSS-VALIDATION RESULTS ===================')
    print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities)*100.,
                        np.std(sensitivities)*100.))
    print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities)*100.,
                        np.std(specificities)*100.))
    print("FAR: %.2f%% (+/- %.2f%%)" % (np.mean(fars)*100.,
                    np.std(fars)*100.))
    print("MDR: %.2f%% (+/- %.2f%%)" % (np.mean(mdrs)*100.,
                    np.std(mdrs)*100.))
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies)*100.,
                        np.std(accuracies)*100.))

def main():
    num_features = 4096
    batch_norm = True
    learning_rate = 0.0001
    mini_batch_size = 16
    weight_0 = 1
    epochs = 10
    use_validation = True
    save_features = False
    save_plots = True
    # After the training stops, use train+validation to train for 1 epoch
    use_val_for_training = True
    val_size = 100
    # Threshold to classify between positive and negative
    threshold = 0.5
    # choose between MLP and LSTM for classification of features
    classification_method = 'LSTM'
    # dataset type, 0 for URFD and 1 for New Datasets (Sisfall, NTU etc)
    dataset_type = 0
    # other values include sisfall, ntu etc
    dataset_name = 'urfd'
    # Name of the experiment
    exp = '{}_lr{}_batchs{}_batchnorm{}_w0_{}'.format(
        dataset_name,
        learning_rate,
        mini_batch_size,
        batch_norm,
        weight_0
    )

    if classification_method not in ['MLP', 'LSTM']:
        raise ValueError("Invalid classification method!")

    train(use_validation, use_val_for_training, num_features, learning_rate,
          epochs, threshold, exp, batch_norm, mini_batch_size, save_plots,
          save_features, classification_method, val_size, weight_0, dataset_type)
    
if __name__ == '__main__':
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        
    main()
