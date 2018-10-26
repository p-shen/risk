import numpy as np
import pandas as pd
import keras.backend as K

import model as mdl
import data_generator as gn


def test_transfer():
    BATCH_SIZE = 20
    shuffle = True

    file_input = "/Users/Peter/Documents/GitHub/capstonedata/all_immune/EvalData.txt"
    label_input = "/Users/Peter/Documents/GitHub/capstonedata/all_immune/immune.genes.eval.txt"
    train_features = gn.readFile(file_input)
    train_labels = gn.readFile(label_input)

    train_steps_gen, train_input_size, train_generator = gn.generator_simple(
        train_features, train_labels, shuffle=shuffle, batch_size=BATCH_SIZE)

    model_file_path = "/Users/Peter/Documents/GitHub/risk/tmp/models_test/type_norm_10_25__21_09/checkpoint.0006.hdf5"
    kmodel = mdl.load_savedmodel(model_file_path, custom_objects={
        'negative_log_partial_likelihood': mdl.negative_log_partial_likelihood})
    kmodel = mdl.model_fn_xfer(
        kmodel, 14, 'mean_squared_error', "relu", learning_rate=0.003)

    X, y = next(train_generator)
    pred = kmodel.predict(X)

    kmodel.fit_generator(
        generator=train_generator,
        steps_per_epoch=1,
        epochs=1)

    X, y = next(train_generator)
    pred2 = kmodel.predict(X)


def test_survival():
    survival_time = np.array(
        [361., 1919., 989., 2329., 3622., 871., 1126., 1431., 669., 791.])
    partial_hazard = np.array([0.36544856, 0.3840349, 0.36263838, 0.36480516,
                               0.36792752, 0.3759846, 0.34084716, 0.35269484, 0.329434, 0.3364767])
    censor = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 1])

    ci = mdl.concordance_metric(survival_time, partial_hazard, censor)
    print(ci)

    surv_model = mdl.load_savedmodel(filepath="/Users/Peter/Documents/GitHub/risk/tmp/models_test/type_norm_10_19__15_40/surv.hdf5", custom_objects={
        'negative_log_partial_likelihood': mdl.negative_log_partial_likelihood}, compile=True)
    surv_model = mdl.compile_model(
        surv_model, 0.003, mdl.negative_log_partial_likelihood)

    filename = "/Users/Peter/Documents/GitHub/risk/data/tcga/EvalData.txt"

    features, labels, _ = gn.processDataLabels(
        filename, batch_by_type=False, normalize=False)
    _, _, gen = gn.generator_survival(features, labels)
    loss, acc = surv_model.evaluate_generator(
        gen,
        steps=10)

    # evaluate CI index for evaluation set
    gen = gn.test_generator()
    hazard_features, surv_labels = next(gen)

    hazard_predict = surv_model.predict(hazard_features)
    ci = mdl.concordance_metric(
        surv_labels[:, 0], hazard_predict, surv_labels[:, 1])

    print(ci)

    neg_likelihood = mdl.negative_log_partial_likelihood(
        surv_labels[:, 1], hazard_predict)

    with K.get_session() as sess:
        print(neg_likelihood.eval())


if __name__ == '__main__':
    DEBUG = True

    test_transfer()
