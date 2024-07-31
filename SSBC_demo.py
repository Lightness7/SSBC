
import os
import cv2
import time
import shap
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.io import loadmat
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import train_test_split
from alive_progress import alive_bar
from imblearn.over_sampling import SMOTE, RandomOverSampler



def read_hsi(dataset_name):

    x = None
    y = None
    class_name = None

    data_path = os.path.join(os.getcwd(), 'Dataset')

    if dataset_name == 'indian_pines':
        x = loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        y = loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        class_name = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
                      'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill',
                      'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                      'Stone-Steel-Towers']

    elif dataset_name == 'houston 2013':
        x = loadmat(os.path.join(data_path, 'Houston.mat'))['Houston']
        y = loadmat(os.path.join(data_path, 'Houston_gt.mat'))['Houston_gt']
        class_name = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees',
                      'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
                      'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis court', 'Running track']

    elif dataset_name == 'KSC':
        x = loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        y = loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
        class_name = ['Scrub', 'Willow swamp', 'Cabbage palm hammock', 'Cabbage palm/oak hammock', 'Slash pine',
                      'Oak/broadleaf hammock', 'Hardwood Swamp', 'Graminoid marsh', 'Spartina marsh', 'Cattail marsh',
                      'Salt marsh', 'Mud flats', 'Water']

    else:
        print("No such dataset is available.")

    print(f"\nHSI cube: {x.shape}")

    class_number = max(np.unique(y))
    print(f"The number of classes: {class_number}")
    data_sort = sorted(Counter(np.array(y.astype(int)).flatten()).items())
    print("Dataset composition:", data_sort)

    return x, y, class_name, class_number


def data_pad_zero(data, window_size):

    patch_length = window_size // 2
    data_padded = np.lib.pad(data, ((patch_length, patch_length), (patch_length, patch_length), (0, 0)), 'constant', constant_values=0)

    return data_padded


def dct_patch(data):

    if data.shape[1] % 2 == 0:

        data_pad = np.zeros([1, data.shape[1]])
        data = np.concatenate((data, data_pad), axis=0)
        data = cv2.dct(data.astype('float'))

    else:

        data_pad_0 = np.zeros([data.shape[0], 1])
        data = np.concatenate((data, data_pad_0), axis=1)
        data_pad_0 = np.zeros([1, data.shape[1]])
        data = np.concatenate((data, data_pad_0), axis=0)

        data = cv2.dct(data.astype('float'))

    return data


def joint_feature(data, label, window_size, r, class_map_output):

    start = time.perf_counter()

    height, width, bands = data.shape

    data_padded = data_pad_zero(data, window_size)

    map_label = None

    if class_map_output == 'yes':
        map_label = -1

    elif class_map_output == 'no':
        map_label = 0

    label = label.reshape(-1)

    spectral_spatial_features = np.zeros([height * width, r, bands])

    with alive_bar(height * width, title="DCT", bar="smooth", spinner="waves", force_tty=True) as bar:
        for k in range(0, height * width):
            if height == width:
                i = k - (k // height) * height
                j = k // height
            elif height != width:
                i = k // width
                j = k - (k // width) * width

            patch = data_padded[i: i + window_size, j: j + window_size, :]

            # Transform a patch cube of w × w × bands to a 2-D matrix of w^2 × bands
            patch_dct_input = patch.reshape(-1, bands)

            if label[k] != map_label:
                patch_after_dct = dct_patch(patch_dct_input)
                patch_after_dct = patch_after_dct[0:r, 0:bands]
                spectral_spatial_features[k, :, :] = patch_after_dct

            else:
                spectral_spatial_features[k, :, :] = np.zeros([r, bands])

            bar()

    spectral_spatial_features = spectral_spatial_features.reshape(-1, r * bands)

    end = time.perf_counter()
    print(f"DCT runtime: {round(end - start)} seconds.\nSpectral spatial features: {spectral_spatial_features.shape[1]}")

    return spectral_spatial_features


def band_correlation_calculation(data, band_range):

    start = time.perf_counter()

    height = data.shape[0]
    width = data.shape[1]
    bands = data.shape[2]
    bands_in_piece_number = bands // band_range
    input_band_index = bands_in_piece_number // 2 + 1  # Band index for calculation
    print(f"Number of bands per band range: {bands_in_piece_number}")

    band_correlation_feature_1 = np.zeros([height, width])
    band_correlation_feature_2 = np.zeros([height, width])

    band_index_i = input_band_index
    band_index_j = input_band_index

    k = 0

    with alive_bar(band_range * (band_range - 1) // 2, title="Band Correlation", bar="smooth", spinner="waves", force_tty=True) as bar:
        for i in range(0, band_range):
            for j in range(i + 1, band_range):

                band_i_index = i * bands_in_piece_number + band_index_i
                band_j_index = j * bands_in_piece_number + band_index_j

                # The input of the calculation is the middle band
                # band_i = data[:, :, band_i_index]
                # band_j = data[:, :, band_j_index]

                # The input of the calculation is the average band
                band_i = np.mean(data[:, :, bands_in_piece_number * i : bands_in_piece_number * (i + 1)], axis=2)
                band_j = np.mean(data[:, :, bands_in_piece_number * j : bands_in_piece_number * (j + 1)], axis=2)

                # The input of the calculation is the max band
                # band_i = np.max(data[:, :, bands_in_piece_number * i : bands_in_piece_number * (i + 1)], axis=2)
                # band_j = np.max(data[:, :, bands_in_piece_number * j : bands_in_piece_number * (j + 1)], axis=2)

                imitation_ndvi = (band_i - band_j) / (band_i + band_j + 0.000000001)
                imitation_io = band_i / (band_j + 0.000000001)

                k += 1

                band_correlation_feature_1 = np.dstack((band_correlation_feature_1, imitation_ndvi))
                band_correlation_feature_2 = np.dstack((band_correlation_feature_2, imitation_io))

                bar()

    band_correlation_feature_1 = band_correlation_feature_1[:, :, 1:]
    band_correlation_feature_2 = band_correlation_feature_2[:, :, 1:]

    band_correlation_feature = np.dstack((band_correlation_feature_1, band_correlation_feature_2))

    end = time.perf_counter()
    print(f"Band correlation calculation runtime: {round(end - start)} seconds.")
    print(f"Band correlation features: {band_range * (band_range - 1)}")

    return band_correlation_feature


def remove_background_pixels(data):

    data_removed = data[data['class'] != 0]
    print(f"\nRemove background pixels (label = 0): {data_removed.iloc[:, :-1].shape}")

    x = data_removed.iloc[:, :-1].values
    y = data_removed.loc[:, 'class'].values - 1

    return x, y


def fusion(x, y, feature_1, feature_2, original_spectral):

    start = time.perf_counter()

    feature_for_class_map = None

    height, width, bands = x.shape

    feature_2 = feature_2.reshape(height * width, -1)

    if original_spectral == 'yes':
        feature_for_class_map = pd.concat([pd.DataFrame(x.reshape(-1, bands)), pd.DataFrame(feature_1), pd.DataFrame(feature_2), pd.DataFrame(y.ravel())], axis=1)
        feature_for_class_map.columns = [f'Feature-{i}' for i in range(1, feature_for_class_map.shape[1])] + ['class']

    elif original_spectral == 'no':
        feature_for_class_map = pd.concat([pd.DataFrame(feature_1), pd.DataFrame(feature_2), pd.DataFrame(y.ravel())], axis=1)
        feature_for_class_map.columns = [f'Feature-{i}' for i in range(1, feature_for_class_map.shape[1])] + ['class']

    feature_removed, y_removed = remove_background_pixels(feature_for_class_map)

    end = time.perf_counter()
    print(f"\nFeature fusion runtime：{round(end - start)} seconds")

    return feature_removed, y_removed, feature_for_class_map


def split_data(X, y, rate, s):

    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=rate, random_state=s, stratify=y)
    # stratify：Stratified random sampling, maintaining original class distribution
    print(f"\nTrain set: {X_train.shape} \nTest set: {X_test.shape}")
    data_sort_train = sorted(Counter(np.array(y_train.astype(int)).flatten()).items())
    print("Train set composition:", data_sort_train)
    data_sort_test = sorted(Counter(np.array(y_test .astype(int)).flatten()).items())
    print("Test set composition:", data_sort_test)

    return X_train, X_test, y_train, y_test


def smote_adaptive(data, label):

    # Adaptive SMOTE
    # Adaptively modify the parameter of K-nearest neighbors due to the number of samples per class
    label = np.array(label.astype(int))
    label = label.flatten()
    data_sort = sorted(Counter(label).items())
    print("Adaptive SMOTE start:\nTrain set composition:", data_sort)
    start = time.perf_counter()

    class_num = max(label) + 1

    class_size_list = []
    for n in range(0, class_num):
        class_size = data_sort[n][1]
        class_size_list.append(class_size)

    smote_size = max(class_size_list)

    for i in range(0, class_num):

        if class_size_list[i] <= 5:

            if class_size_list[i] == 1:
                ROS = RandomOverSampler(sampling_strategy={i: smote_size}, random_state=7)
                data, label = ROS.fit_resample(data, label)

            else:
                kn_num = class_size_list[i] - 1
                smote = SMOTE(sampling_strategy={i: smote_size}, random_state=7, k_neighbors=kn_num)
                data, label = smote.fit_resample(data, label)
                # print(f"\nTrain set composition: {sorted(Counter(label).items())}")

        else:
            smote = SMOTE(sampling_strategy={i: smote_size}, random_state=7, k_neighbors=5)
            data, label = smote.fit_resample(data, label)
            # print(f"\nTrain set composition: {sorted(Counter(label).items())}")

    print(f"Train set after Adaptive SMOTE: {data.shape}, \ncomposition: {sorted(Counter(label).items())}")
    end = time.perf_counter()
    print(f"Adaptive SMOTE over, runtime: {round(end - start)} seconds.")

    return data, label


def lgbm(x_train, y_train, x_test, y_test):

    class_number = max(np.unique(y_train)) + 1

    params = {
        'task': 'train',
        'boosting_type': 'goss',
        'objective': 'multiclass',
        'num_class': class_number,
        'metric': 'multi_logloss',
        'max_depth': 4,
        'num_leaves': 20,
        'learning_rate': 0.02,
        'min_data_in_leaf': 60,
        'feature_fraction': 0.05,
        'verbose': -1
    }

    boost_round = 1000

    print("LightGBM:")
    start = time.perf_counter()

    x_train, y_train = smote_adaptive(x_train, y_train)

    lgb_trainset = lgb.Dataset(x_train, y_train)
    model = lgb.train(params, lgb_trainset, num_boost_round=boost_round)
    print("LightGBM done.")

    ypred_proba = model.predict(x_test)
    ypred = [list(x).index(max(x)) for x in ypred_proba]

    end = time.perf_counter()

    print(f"LightGBM runtime：{round(end - start)} seconds.")

    # Calculate the accuracy per class
    ypred_class = pd.DataFrame(ypred)
    ypred_class = np.array(ypred_class)
    class_accuracies = []
    for class_ in np.unique(y_test):
        class_acc = np.mean(ypred_class[y_test == class_] == class_)
        class_accuracies.append(class_acc)
    print(f'Class Accuracies: \n{class_accuracies}')

    return ypred_proba, ypred, model, class_accuracies


def testset_report(y_test, ypred, class_name):

    print("\nTest set report:")
    print(metrics.classification_report(y_test, ypred, digits=4, target_names=class_name))
    y_test = pd.DataFrame(data=y_test)
    ypred = pd.DataFrame(data=ypred)
    print('Cohen Kappa score: %.4f' % metrics.cohen_kappa_score(y_test, ypred))

    return


def shap_explain(X_train, X_test, model):

    shap.initjs()

    X_train = pd.DataFrame(X_train)
    X_train.columns = [f'OSB-{i}' for i in range(1, 201)] + [f'JSS-{i}' for i in range(1, 1001)] + [f'BC-{i}' for i in range(1, 651)]
    X_test = pd.DataFrame(X_test)
    X_test.columns = [f'OSB-{i}' for i in range(1, 201)] + [f'JSS-{i}' for i in range(1, 1001)] + [f'BC-{i}' for i in range(1, 651)]
    Feature_name = [f'OSB-{i}' for i in range(1, 201)] + [f'JSS-{i}' for i in range(1, 1001)] + [f'BC-{i}' for i in range(1, 651)]
    Class_name = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
                   'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill',
                   'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                   'Stone-Steel-Towers']

    # explain the model's predictions using SHAP values
    # (same syntax works for LightGBM, CatBoost, and scikit-learn models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig = plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['figure.dpi'] = 300

    # Summary plot
    shap.summary_plot(shap_values, X_test.values, feature_names=Feature_name, class_names=Class_name, max_display=30, show=False)
    plt.tight_layout()
    plt.show()

    return



if __name__ == "__main__":

    start = time.perf_counter()

    #                     0           1          2        #
    dataset_list = ['indian_pines', 'KSC', 'houston 2013']

    dataset_name = dataset_list[0]

    #          0      1   #
    option = ['no', 'yes']
    original_spectral = option[1]  # Whether to involve original spectral bands in the features
    class_map_output = option[0]  # Whether to generate a classification map, implemented only if experiment_time is set to 1
    shap_output = option[0]  # Whether to generate a summary plot of SHAP, implemented only if experiment_time is set to 1

    # Times of experiments
    experiment_time = 1  # 1 experiment
    # experiment_time = 10  # The average of 10 experiments

    # Parameters
    window_size = 27  # Size of local window
    r = 5  # Retain the first r rows of the low-frequency components in the matrix after DCT, range: 1-window_size
    band_range = 26  # Number of groups in the calculation of band correlation features

    test_size = 0.90  # Proportion of test set divided

    # Load data
    x, y, class_name, class_number = read_hsi(dataset_name)

    height, width, bands = x.shape

    # Joint spectral-spatial feature extraction
    spectral_spatial_features = joint_feature(x, y, window_size, r, class_map_output)

    # Calculation of band correlation features
    band_features = band_correlation_calculation(x, band_range)

    # Feature fusion
    x_removed, y_removed, data_for_class_map = fusion(x, y, spectral_spatial_features, band_features, original_spectral)

    # Classification phase
    if experiment_time == 1:

        # Split training and test sets
        x_train, x_test, y_train, y_test = split_data(x_removed, y_removed, test_size, 7)

        # Training
        ypred_proba, ypred, model, class_accuracies = lgbm(x_train, y_train, x_test, y_test)

        # Report predicted results
        print(testset_report(y_test, ypred, class_name))

        oa_all = metrics.accuracy_score(y_test, ypred)
        aa = np.mean(class_accuracies)
        kappa = metrics.cohen_kappa_score(y_test, ypred)

        class_accuracies_100 = []
        for accs in class_accuracies:
            accs_100 = str(round(accs * 100, 2))
            class_accuracies_100.append(accs_100)

        if shap_output == 'yes':
            # SHAP: model explanation
            print("\nMapping summary plot of SHAP:")
            print(shap_explain(x_train, x_test, model))

        if class_map_output == 'yes':

            print("\nMapping classification map:")
            start = time.perf_counter()

            # Generate classification maps
            dataset_size = data_for_class_map.shape[0]
            ypred_proba = model.predict(data_for_class_map.iloc[:, :-1].values.reshape(dataset_size, -1))
            l = [list(x).index(max(x)) for x in ypred_proba]

            end = time.perf_counter()
            print(f"\nPredictions over, runtime: {round(end - start)} seconds.")

            data_predicted = np.array(l).reshape((height, width))
            data_predicted = pd.DataFrame(data_predicted)

            fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8),
                                      # dpi=300
                                      )

            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 5
            # plt.rcParams['font.weight'] = 'bold'

            ax1.imshow(data_predicted, cmap='gist_rainbow', interpolation='none')  # interpolation='none'：避免分类图出现花边
            ax1.axes.xaxis.set_visible(False)
            ax1.axes.yaxis.set_visible(False)

            plt.show()

    if experiment_time == 10:

        seed = [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]

        oa_all = []
        aa_all = []
        kappa_all = []
        each_acc_all = np.zeros([len(seed), class_number])

        run_time = 0

        for s in seed:
            # Split training and test sets
            x_train, x_test, y_train, y_test = split_data(x_removed, y_removed, test_size, s)

            # Training
            ypred_proba, ypred, model, class_accuracies = lgbm(x_train, y_train, x_test, y_test)

            # Report predicted results
            print(testset_report(y_test, ypred, class_name))

            oa = metrics.accuracy_score(y_test, ypred)
            aa = np.mean(class_accuracies)
            kappa = metrics.cohen_kappa_score(y_test, ypred)

            oa_all.append(oa)
            aa_all.append(aa)
            kappa_all.append(kappa)
            each_acc_all[run_time, :] = class_accuracies

            run_time += 1

        # variance
        oa_var = np.var(oa_all)
        aa_var = np.var(aa_all)
        kappa_var = np.var(kappa_all)

        # standard deviation
        oa_std = np.std(oa_all, ddof=1)
        aa_std = np.std(aa_all, ddof=1)
        kappa_std = np.std(kappa_all, ddof=1)

        print(f"OA: mean {np.mean(oa_all)}, min {min(oa_all)}, max {max(oa_all)}, var {oa_var}, std {oa_std}")
        print(f"AA: mean {np.mean(aa_all)}, min {min(aa_all)}, max {max(aa_all)}, var {aa_var}, std {aa_std}")
        print(f"Kappa: mean {np.mean(kappa_all)}, min {min(kappa_all)}, max {max(kappa_all)}, var {kappa_var}, std {kappa_std}")

    end = time.perf_counter()
    print(f"\nTotal runtime：{round(end - start)} seconds")

