""" sheet4_tests.py

(c) Felix Brockherde, TU Berlin, 2013-2016
"""
import unittest

import numpy as np
import sheet4 as imp

class TestSheet4(unittest.TestCase):
    def test_compute_box_constraints_equal_y(self):
        C = imp.svm_smo(kernel='linear', C=1.)
        res1 = C._compute_box_constraints(0, 1, [-1, -1, 1, 1], np.array([1.0, 0.5, 0.01, 0.2]), C=1.0)
        res2 = C._compute_box_constraints(0, 1, [-1, -1, 1, 1], np.array([1.0, 0.5, 0.01, 0.2]), C=10.0)
        res3 = C._compute_box_constraints(0, 1, [-1, -1, 1, 1], np.array([1.0, 0.5, 0.01, 0.2]), C=1.0)
        self.assertTrue((np.allclose(res1, [0.5, 1.0]) and
                         np.allclose(res2, [0.0, 1.5]) and
                         np.allclose(res3, [0.5, 1.0])),
                        msg='_compute_box_constraints: Error in y_i == y_j case.')

    def test_compute_box_constraints_different_y(self):
        C = imp.svm_smo(kernel='linear', C=1.)
        res4 = C._compute_box_constraints(1, 2, [-1, -1, 1, 1], np.array([1.0, 0.5, 0.01, 0.2]), C=1.0)
        res5 = C._compute_box_constraints(1, 3, [-1, -1, 1, 1], np.array([0.01, 0.5, 0.01, 4.2]), C=1.0)
        self.assertTrue((np.allclose(res4, [0.0, 0.51], atol=1e-3) and
                            np.allclose(res5, [3.7, 1.0], atol=1e-3)),
                        msg='_compute_box_constraints: Error in y_i != y_j case.')

    def test_compute_updated_b(self):
        C = imp.svm_smo(kernel='linear', C=1.)
        res1 = C._compute_updated_b(1.0, 2.0, 0, 1, np.array([[1.0, 0.3], [0.3, 1.0]]), [-1., +1.], [1., 0.5], [1., -1.5], 0.3, 1.2)
        res2 = C._compute_updated_b(1.0, 2.0, 0, 1, np.array([[1.0, 0.3], [0.3, 1.0]]), [-1., +1.], [1., 0.5], [2., 0.5], 0.3, 1.2)
        res3 = C._compute_updated_b(1.0, 2.0, 0, 1, np.array([[1.0, 0.3], [0.3, 1.0]]), [-1., +1.], [1., 0.5], [2., -1.5], 0.3, 1.2)
        self.assertTrue((np.allclose([res1, res2, res3], [0.7, 2.0, -0.15], atol=1e-3)),
                        msg='_compute_updated_b: Error found.')

    def test_update_parameters_kappa(self):
        C = imp.svm_smo(kernel='linear', C=1.)
        res1 = C._update_parameters(1.0, 0.5, 0, 1, np.array([[1.0, 1.6], [1.6, 1.0]]), np.array([-1, +1]), np.array([0.4, -0.5]), 1.2, 0.1)
        self.assertTrue(res1[2] == False, msg='_update_parameters Error in kappa condition.')

    def test_update_parameters(self):
        C = imp.svm_smo(kernel='linear', C=1.)
        res2 = C._update_parameters(1.0, 0.5, 0, 1, np.array([[1.0, 0.3], [0.3, 1.0]]), np.array([-1, +1]), np.array([0.4, -0.5]), 1.2, 0.1)
        res3 = C._update_parameters(1.0, 2.0, 0, 1, np.array([[1.0, 0.3], [0.3, 1.0]]), np.array([-1, +1]), np.array([0.4, -0.5]), 1.2, 0.1)
        self.assertTrue(np.allclose(res2[0], [0.1, -0.8]), msg=f"{res2}")
        self.assertTrue(np.allclose(res2[1], 2.41, atol=1e-3), msg=f"{res2}")
        self.assertTrue(np.allclose(res3[0], [0.9, 0.0]), msg=f"{res3}")
        self.assertTrue(np.allclose(res3[1], 2.7, atol=1e-3), msg=f"{res3}")

    def test_svm_smo(self):
        np.random.seed(1)
        X_tr = np.hstack((np.random.normal(size=[2, 30]), np.random.normal(size=[2, 30]) + np.array([2., 2.])[:, np.newaxis])).T
        Y_tr = np.array([1] * 30 + [-1] * 30)
        X_te = np.hstack((np.random.normal(size=[2, 30]), np.random.normal(size=[2, 30]) + np.array([2., 2.])[:, np.newaxis])).T
        Y_te = np.array([1] * 30 + [-1] * 30)
        C = imp.svm_smo(kernel='linear', C=1.)
        C.fit(X=X_tr, Y=Y_tr)
        Y_pred = C.predict(X_te)
        loss = float(np.sum(np.sign(Y_te) != np.sign(Y_pred)))/float(len(Y_te))
        imp.plot_svm_2d(X_tr, Y_tr, C)
        print('test case loss', loss)
        self.assertTrue(loss < 0.25, msg='svm_smo: Error. The loss is %.2f and should be below 0.25' % loss)

    def test_svm_qp(self):
        C = imp.svm_qp(kernel='linear', C=1.)
        np.random.seed(1)
        X_tr = np.hstack((np.random.normal(size=[2, 30]), np.random.normal(size=[2, 30]) + np.array([2., 2.])[:, np.newaxis])).T
        Y_tr = np.array([1] * 30 + [-1] * 30)
        X_te = np.hstack((np.random.normal(size=[2, 30]), np.random.normal(size=[2, 30]) + np.array([2., 2.])[:, np.newaxis])).T
        Y_te = np.array([1] * 30 + [-1] * 30)
        C.fit(X=X_tr, Y=Y_tr)
        Y_pred = C.predict(X_te)
        loss = float(np.sum(np.sign(Y_te) != np.sign(Y_pred)))/float(len(Y_te))
        imp.plot_svm_2d(X_tr, Y_tr, C)
        print('test case loss', loss)
        self.assertTrue(loss < 0.25, msg=f'svm_qp: Error. The loss is {loss:.2f} and should be below 0.25')

if __name__ == '__main__':
    unittest.main()
