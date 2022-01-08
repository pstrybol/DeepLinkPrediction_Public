from DeepLinkPrediction.utils import *


def test_generate_traintest_dependencies(dependency_df, heterogenous_nw_obj):
    threshold_neg = -0.5
    threshold_pos = -1.5
    npr = 5
    gene2int = heterogenous_nw_obj.gene2int

    # First no train_ratio and no train_validation_ratio
    train_ratio = None
    train_validaiton_ratio = None

    negs, negs_arr, pos, pos_arr, intermediate, intermediate_arr = generate_traintest_dependencies(
        dependency_df,
        threshold_neg=threshold_neg,
        threshold_pos=threshold_pos,
        npr=npr,
        gene2int=gene2int,
        train_test_ratio=train_ratio,
        train_validaiton_ratio=train_validaiton_ratio)

    assert negs_arr.shape[0] == np.sum([len(g) for g in negs.values()]), "ERROR SHAPE MISMATCH"
    assert pos_arr.shape[0] == np.sum([len(g) for g in pos.values()]), "ERROR SHAPE MISMATCH"
    assert intermediate_arr.shape[0] == np.sum([len(g) for g in intermediate.values()]), "ERROR SHAPE MISMATCH"
    for cl, g in pos.items():
        assert not set(g) & set(negs[cl])
        assert not set(g) & set(intermediate[cl])

    del negs, negs_arr, pos, pos_arr, intermediate, intermediate_arr, train_ratio, train_validaiton_ratio

    # Second with train_ratio and no train_validation_ratio
    train_ratio = 0.8
    train_validaiton_ratio = None

    negs, negs_arr_train, negs_arr_test, pos, pos_arr_train, pos_arr_test,\
        intermediate, intermediate_arr_train, intermediate_arr_test = generate_traintest_dependencies(
            dependency_df,
            threshold_neg=threshold_neg,
            threshold_pos=threshold_pos,
            npr=npr,
            gene2int=gene2int,
            train_test_ratio=train_ratio,
            train_validaiton_ratio=train_validaiton_ratio)

    assert not set(map(tuple, negs_arr_train)) & set(map(tuple, negs_arr_test)), "Overlap negatives train - test"
    assert not set(map(tuple, intermediate_arr_train)) & set(map(tuple, intermediate_arr_test)), "Overlap intermediates train - test"
    assert not set(map(tuple, pos_arr_train)) & set(map(tuple, pos_arr_test)), "Overlap positives train - test"

    assert negs_arr_train.shape[0] == len(set(map(tuple, negs_arr_train))),"Duplicate negative train interactions"
    assert intermediate_arr_train.shape[0] == len(set(map(tuple, intermediate_arr_train))),"Duplicate intermediary train interactions"
    assert pos_arr_train.shape[0] == len(set(map(tuple, pos_arr_train))),"Duplicate positive train interactions"

    assert negs_arr_test.shape[0] == len(set(map(tuple, negs_arr_test))), "Duplicate negative test interactions"
    assert intermediate_arr_test.shape[0] == len(
        set(map(tuple, intermediate_arr_test))), "Duplicate intermediary test interactions"
    assert pos_arr_test.shape[0] == len(set(map(tuple, pos_arr_test))), "Duplicate positive test interactions"


    assert negs_arr_train.shape[0] + negs_arr_test.shape[0] == np.sum(
        [len(g) for g in negs.values()]), "ERROR SHAPE MISMATCH"
    assert pos_arr_train.shape[0] + pos_arr_test.shape[0] == np.sum(
        [len(g) for g in pos.values()]), "ERROR SHAPE MISMATCH"
    assert intermediate_arr_train.shape[0] + intermediate_arr_test.shape[0] == np.sum(
        [len(g) for g in intermediate.values()]), "ERROR SHAPE MISMATCH"
    for i, arr in enumerate([negs_arr_train, negs_arr_test, pos_arr_train, pos_arr_test, intermediate_arr_train,
                             intermediate_arr_test]):
        if i < 2:
            assert len(
                set([gene2int[i] for i in negs.keys()]) - set(np.unique(arr))) == 0, 'ERROR CELL LINES LOST IN SPLIT'
        if i > 1 and i < 5:
            assert len(
                set([gene2int[i] for i in pos.keys()]) - set(np.unique(arr))) == 0, 'ERROR CELL LINES LOST IN SPLIT'
        if i > 4:
            assert len(set([gene2int[i] for i in intermediate.keys()]) - set(
                np.unique(arr))) == 0, 'ERROR CELL LINES LOST IN SPLIT'
    for cl, g in pos.items():
        assert not set(g) & set(negs[cl])
        assert not set(g) & set(intermediate[cl])

    del negs, negs_arr_train, negs_arr_test, pos, pos_arr_train, pos_arr_test, intermediate,\
        intermediate_arr_train, intermediate_arr_test, train_ratio, train_validaiton_ratio

    # Third with train_ratio and train_validation_ratio
    train_ratio = 0.8
    train_validaiton_ratio = 0.8

    negs, negs_arr_train, negs_arr_val, negs_arr_test, pos, pos_arr_train, pos_arr_val, pos_arr_test, \
    intermediate, intermediate_arr_train, intermediate_arr_val, intermediate_arr_test = generate_traintest_dependencies(
        dependency_df,
        threshold_neg=threshold_neg,
        threshold_pos=threshold_pos,
        npr=npr,
        gene2int=gene2int,
        train_test_ratio=train_ratio,
        train_validaiton_ratio=train_validaiton_ratio)

    assert negs_arr_train.shape[0] == len(set(map(tuple, negs_arr_train))), "Duplicate negative train interactions"
    assert intermediate_arr_train.shape[0] == len(
        set(map(tuple, intermediate_arr_train))), "Duplicate intermediary train interactions"
    assert pos_arr_train.shape[0] == len(set(map(tuple, pos_arr_train))), "Duplicate positive train interactions"

    assert negs_arr_val.shape[0] == len(set(map(tuple, negs_arr_val))), "Duplicate negative val interactions"
    assert intermediate_arr_val.shape[0] == len(
        set(map(tuple, intermediate_arr_val))), "Duplicate intermediary val interactions"
    assert pos_arr_val.shape[0] == len(set(map(tuple, pos_arr_val))), "Duplicate positive train interactions"

    assert pos_arr_test.shape[0] == len(set(map(tuple, pos_arr_test))), "Duplicate positive test interactions"
    assert negs_arr_test.shape[0] == len(set(map(tuple, negs_arr_test))), "Duplicate negative test interactions"
    assert intermediate_arr_test.shape[0] == len(
        set(map(tuple, intermediate_arr_test))), "Duplicate intermediary test interactions"
    assert pos_arr_test.shape[0] == len(set(map(tuple, pos_arr_test))), "Duplicate positive test interactions"

    assert not set(map(tuple, negs_arr_train)) & set(map(tuple, negs_arr_test)), "Overlap negatives train - test"
    assert not set(map(tuple, intermediate_arr_train)) & set(
        map(tuple, intermediate_arr_test)), "Overlap intermediates train - test"
    assert not set(map(tuple, pos_arr_train)) & set(map(tuple, pos_arr_test)), "Overlap positives train - test"

    assert not set(map(tuple, negs_arr_val)) & set(map(tuple, negs_arr_test)), "Overlap negatives val - test"
    assert not set(map(tuple, intermediate_arr_val)) & set(
        map(tuple, intermediate_arr_test)), "Overlap intermediates val - test"
    assert not set(map(tuple, pos_arr_val)) & set(map(tuple, pos_arr_test)), "Overlap positives val - test"



    assert negs_arr_train.shape[0] + negs_arr_val.shape[0] + negs_arr_test.shape[0] == np.sum(
        [len(g) for g in negs.values()]), "ERROR SHAPE MISMATCH"
    assert pos_arr_train.shape[0] + pos_arr_val.shape[0] + pos_arr_test.shape[0] == np.sum(
        [len(g) for g in pos.values()]), "ERROR SHAPE MISMATCH"
    assert intermediate_arr_train.shape[0] + intermediate_arr_val.shape[0] + intermediate_arr_test.shape[0] == np.sum(
        [len(g) for g in intermediate.values()]), "ERROR SHAPE MISMATCH"

    for i, arr in enumerate([negs_arr_train, negs_arr_val, negs_arr_test, pos_arr_train, pos_arr_val, pos_arr_test,
                             intermediate_arr_train, intermediate_arr_val, intermediate_arr_test]):
        if i < 3:
            assert len(
                set([gene2int[i] for i in negs.keys()]) - set(np.unique(arr))) == 0, 'ERROR CELL LINES LOST IN SPLIT'
        if i > 2 and i < 6:
            assert len(
                set([gene2int[i] for i in pos.keys()]) - set(np.unique(arr))) == 0, 'ERROR CELL LINES LOST IN SPLIT'
        if i > 5:
            assert len(set([gene2int[i] for i in intermediate.keys()]) - set(
                np.unique(arr))) == 0, 'ERROR CELL LINES LOST IN SPLIT'
    for cl, g in pos.items():
        assert not set(g) & set(negs[cl])
        assert not set(g) & set(intermediate[cl])


def test_extract_x_dict_at_threshold(dependency_df, heterogenous_nw_obj):

    threshold_neg = -0.5
    threshold_pos = -1.5
    npr_dep = 3
    gene2int = heterogenous_nw_obj.gene2int

    pos_e = extract_pos_dict_at_threshold(dependency_df, threshold=threshold_pos)
    interm_e = extract_interm_dict_at_threshold(dependency_df, pos_dict=pos_e, neg_threshold=threshold_neg,
                                                pos_threshold=threshold_pos)

    # First no train_ratio and no train_validation_ratio
    train_ratio = None
    train_validaiton_ratio = None
    negs, negs_arr, pos, pos_arr, intermediate, intermediate_arr = generate_traintest_dependencies(
        dependency_df,
        threshold_neg=threshold_neg,
        threshold_pos=threshold_pos,
        npr=npr_dep,
        gene2int=gene2int,
        train_test_ratio=train_ratio,
        train_validaiton_ratio=train_validaiton_ratio)

    # Don't test negs_e == negs because the negative are chosen at random 5x the number positives for each cell line
    check_equality_dict(pos, pos_e)
    check_equality_dict(intermediate, interm_e)

    del negs, negs_arr, pos, pos_arr, intermediate, intermediate_arr, train_ratio, train_validaiton_ratio

    # Second with train_ratio and no train_validation_ratio
    train_ratio = 0.8
    train_validaiton_ratio = None

    negs, negs_arr_train, negs_arr_test, pos, pos_arr_train, pos_arr_test, \
    intermediate, intermediate_arr_train, intermediate_arr_test = generate_traintest_dependencies(
        dependency_df,
        threshold_neg=threshold_neg,
        threshold_pos=threshold_pos,
        npr=npr_dep,
        gene2int=gene2int,
        train_test_ratio=train_ratio,
        train_validaiton_ratio=train_validaiton_ratio)

    check_equality_dict(pos, pos_e)
    check_equality_dict(intermediate, interm_e)

    del negs, negs_arr_train, negs_arr_test, pos, pos_arr_train, pos_arr_test, intermediate, \
        intermediate_arr_train, intermediate_arr_test, train_ratio, train_validaiton_ratio

    # Third with train_ratio and train_validation_ratio
    train_ratio = 0.8
    train_validaiton_ratio = 0.9

    negs, negs_arr_train, negs_arr_val, negs_arr_test, pos, pos_arr_train, pos_arr_val, pos_arr_test, \
    intermediate, intermediate_arr_train, intermediate_arr_val, intermediate_arr_test = generate_traintest_dependencies(
        dependency_df,
        threshold_neg=threshold_neg,
        threshold_pos=threshold_pos,
        npr=npr_dep,
        gene2int=gene2int,
        train_test_ratio=train_ratio,
        train_validaiton_ratio=train_validaiton_ratio)

    check_equality_dict(pos, pos_e)
    check_equality_dict(intermediate, interm_e)


def test_construct_cellline_splits_all(dependency_df, heterogenous_nw_obj):
    gene2int = heterogenous_nw_obj.gene2int
    deps='intermpos'
    threshold_neg = -0.5
    threshold_pos = -1.5
    npr = 5
    train_ratio = 0.8
    train_validaiton_ratio = 0.9

    negs, negs_arr_train, negs_arr_val, negs_arr_test, pos, pos_arr_train, pos_arr_val, pos_arr_test, \
    intermediate, intermediate_arr_train, intermediate_arr_val, intermediate_arr_test = generate_traintest_dependencies(
        dependency_df,
        threshold_neg=threshold_neg,
        threshold_pos=threshold_pos,
        npr=npr,
        gene2int=gene2int,
        train_test_ratio=train_ratio,
        train_validaiton_ratio=train_validaiton_ratio)

    out_d = construct_cellline_splits_all(intermediate, pos, gene2int, deps=deps, fp=None, return_d=True)

    assert True

def test_read_write_h5py(h5py_fp):
    # TODO: add write and compare after reading if matrices are the same
    read_h5py(h5py_fp, dtype=int)
