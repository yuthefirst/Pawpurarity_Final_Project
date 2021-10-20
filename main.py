# Import libraries
import gc
import numpy as np
import pandas as pd
import random
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.layers import Input
import cv2
#######################################################################################################################
# Constants
IMG_SIZE = 384
CHANNELS = 3
BATCH_SIZE = 16
Q = 30
EPOCHS = 10
# FOLDS = 5
# FEATURE_FOLDS = 5
FOLDS = 2
FEATURE_FOLDS = 3
SEED = 42
VERBOSE = 1
LR = 0.000005

# Logic...
TRAIN_FEATURE_MODEL = False

# Folders
DATA_DIR = 'petfinder-pawpularity-score/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'
AUTOTUNE = tf.data.experimental.AUTOTUNE

#######################################################################################################################

# Load Train Data
train_df = pd.read_csv(f'{DATA_DIR}train.csv')
train_df['Id'] = train_df['Id'].apply(lambda x: f'{TRAIN_DIR}{x}.jpg')

# Set a specific label to be able to perform stratification
train_df['stratify_label'] = pd.qcut(train_df['Pawpularity'], q = Q, labels = range(Q))

# Label value to be used for feature model 'classification' training.
train_df['target_value'] = train_df['Pawpularity'] / 100.

#######################################################################################################################

# Load Test Data
test_df = pd.read_csv(f'{DATA_DIR}test.csv')
test_df['Id'] = test_df['Id'].apply(lambda x: f'{TEST_DIR}{x}.jpg')
test_df['Pawpularity'] = 0

#######################################################################################################################
# Load Test Data after training
# df_best_score = pd.DataFrame()
#
# df_best_score = train_df[['Id','Pawpularity']][(train_df['Pawpularity'] <= 100)
#           & (train_df['Pawpularity'] >= 80)]
#
# df_least_score = train_df[['Id','Pawpularity']][(train_df['Pawpularity'] <= 20)
#           & (train_df['Pawpularity'] >= 0)]
#
# test_df_1 = df_best_score.sample(n=10,replace=True)
# print(test_df_1.head())
# test_df_2 = test_df_1.copy()
# test_df_2['Pawpularity'] = 0
# print(test_df_2.head())

#######################################################################################################################
# Working function
def build_augmenter(is_labelled):
    def augment(img):
        # Only use basic augmentations...too much augmentation hurts performance
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_saturation(img, 0.95, 1.05)
        img = tf.image.random_brightness(img, 0.05)
        img = tf.image.random_contrast(img, 0.95, 1.05)
        img = tf.image.random_hue(img, 0.05)

        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if is_labelled else augment


def build_decoder(is_labelled):
    def decode(path):
        # Read Image
        file_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(file_bytes, channels=CHANNELS)

        # Normalize and Resize
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if is_labelled else decode


def create_dataset(df, batch_size=32, is_labelled=False, augment=False, repeat=False, shuffle=False):
    decode_fn = build_decoder(is_labelled)
    augmenter_fn = build_augmenter(is_labelled)

    # Create Dataset
    if is_labelled:
        dataset = tf.data.Dataset.from_tensor_slices((df['Id'].values, df['target_value'].values))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((df['Id'].values))
    dataset = dataset.map(decode_fn, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(augmenter_fn, num_parallel_calls=AUTOTUNE) if augment else dataset
    dataset = dataset.repeat() if repeat else dataset
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True) if shuffle else dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

#######################################################################################################################
# Set Callbacks
def model_checkpoint(fold):
    return tf.keras.callbacks.ModelCheckpoint(
        f'feature_model/feature_model_{fold}.h5',
        verbose=1,
        monitor='val_rmse',
        mode='min',
        save_weights_only=True,
        save_best_only=True)


def unfreeze_model(model):
    # Unfreeze layers while leaving BatchNorm layers frozen
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False


def create_model():
    # Create and Compile Model and show Summary
    effnet_model = efn.EfficientNetB2(include_top=False,
                                      classes=None,
                                      input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS),
                                      weights='feature_model/efficientnet-b2_noisy-student_notop.h5',
                                      pooling='avg')

    # Set all layers to Trainable except BN layers
    unfreeze_model(effnet_model)

    X = tf.keras.layers.Dropout(0.25)(effnet_model.output)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(X)

    # Create Final Model
    model = tf.keras.Model(inputs=effnet_model.input, outputs=output)

    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError('rmse')])

    return model

#######################################################################################################################



#######################################################################################################################
def Pawpularity_Caculation(test_df_2, num_image):
    # Loop through all Feature Extraction Models
    SEED = 42
    FOLDS = 2
    FEATURE_FOLDS = 3
    train_df_1 = train_df.sample(n=num_image, replace=True)
    # Placeholders
    preds_final = np.zeros((test_df_2.shape[0], 1))
    all_oof_score = []
    # Stratification and Label values
    Y_strat = train_df_1['stratify_label'].values
    Y_pawpularity = train_df_1['Pawpularity'].values
    for fold_index in range(FEATURE_FOLDS):
        print("")
        print(f'===== Run for Feature Model {fold_index} \n')

        # Pre model.fit cleanup
        tf.keras.backend.clear_session()
        gc.collect()

        # Create Model
        model = create_model()

        # Load Weights...Use the provided weight files...or modify for your own set.
        model.load_weights(f'feature_model/feature_model_{fold_index}.h5')

        # Strip Last layers to be able to extract features
        model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

        # Summary...only on first load
        # if fold_index == 0: print(model.summary())

        # Feature Extraction
        print('\n===== Extracting Features')
        cb_train_set = create_dataset(
            # train_df,
            train_df_1,
            batch_size=BATCH_SIZE,
            is_labelled=True,
            augment=False,
            repeat=False,
            shuffle=False)
        cb_test_set = create_dataset(
            # test_df,
            test_df_2,
            batch_size=BATCH_SIZE,
            is_labelled=False,
            augment=False,
            repeat=False,
            shuffle=False)

        cb_train_features = model.predict(cb_train_set, verbose=VERBOSE)

        cb_test_features = model.predict(cb_test_set, verbose=VERBOSE)

        print('\n===== Feature Set Shapes')
        print(f'Train Feature Set Shape: {cb_train_features.shape}')
        print(f'Test Feature Set Shape: {cb_test_features.shape}')

        # Stratified Training for CatBoost
        print(f'\n===== Running XGBRegressor - SEED {SEED}')

        # Placeholders
        oof_score = 0

        kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        for idx, (train, val) in enumerate(kfold.split(cb_train_features, Y_strat)):
            print(f'\n===== XGBRegressor Fold {idx} ')

            train_x, train_y = cb_train_features[train], Y_pawpularity[train]
            val_x, val_y = cb_train_features[val], Y_pawpularity[val]

            # Set XGBRegressor Parameters
            cb_params = {'loss_function': 'RMSE',
                         'eval_metric': 'rmse',
                         'iterations': 1000,
                         'grow_policy': 'lossguide',
                         'depth': 6,
                         'l2_leaf_reg': 2.0,
                         'random_strength': 1.0,
                         'learning_rate': 0.05,
                         'task_type': 'GPU',
                         'devices': '0',
                         'verbose': 0,
                         'random_state': SEED}

            # Create and Fit XGBRegressor Model
            cb_model = XGBRegressor(**cb_params)

            cb_model.fit(train_x, train_y, eval_set=[(val_x, val_y)], early_stopping_rounds=100, verbose=250)

            y_pred = cb_model.predict(val_x)

            preds_final += np.array([cb_model.predict(cb_test_features)]).T

            # Update OOF Score
            oof_score += np.sqrt(mean_squared_error(val_y, y_pred))

            # Cleanup
            del cb_model, y_pred
            del train_x, train_y
            del val_x, val_y
            gc.collect()

            # OOF Score for XGBRegressor run
        oof_score /= FOLDS
        all_oof_score.append(oof_score)
        print(f'XGBRegressor OOF Score: {oof_score}')
        print('Test Predictions Cumulative...')
        print(preds_final)

        # Increase to improve randomness on the next feature model run
        SEED += 1

    preds_final /= (FOLDS * FEATURE_FOLDS)
    final_all_oof_score = np.mean(all_oof_score)
    print(f'Final Out Of Fold RMSE Score for all feature models: {final_all_oof_score}')
    return preds_final, final_all_oof_score
#######################################################################################################################









