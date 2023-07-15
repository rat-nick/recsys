from utils.tensor import *
import pytest


def test_split_1d(tensor1d):
    original = tensor1d
    nz = original.count_nonzero()
    train, test = split(original)
    train_nz = train.count_nonzero()
    test_nz = test.count_nonzero()
    assert train_nz + test_nz == nz
    assert train_nz > 0
    assert test_nz > 0
    assert torch.equal(train + test, original)
    assert train_nz / test_nz == pytest.approx(4)


def test_split_2d(tensor2d):
    original = tensor2d
    nz = original.count_nonzero()
    train, test = split(original)
    train_nz = train.count_nonzero()
    test_nz = test.count_nonzero()
    assert train_nz + test_nz == nz
    assert train_nz > 0
    assert test_nz > 0
    assert torch.equal(train + test, original)
    assert train_nz / test_nz == pytest.approx(4)
