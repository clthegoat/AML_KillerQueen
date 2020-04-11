# Task 4

How to run
=======================

1. Generate npy from csv: csv2npy.py
2. Generate fft features with prepare_fft_feature.py
3. Preprocess(Normalization and concat adjacent epoch feature) with preprocessing_ffts
4. Model
4.1. Do cv with cnn_cv_single_epoch.py or cnn_cv_multi_epoch.py
4.2. Make prediction with cnn_pred_single_epoch.py or cnn_pred_multi_epoch.py


Current stage: CNN based model
==========================================

Main schedual:
==============
<table>
  <thead>
    <tr>
      <th>Stage 1</th>
      <th>Stage 2</th>
      <th>Stage 3</th>
      <th>Stage 4</th>
      <th>Stage 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Visualization & Statistics</td>
      <td>Preprocessing & feature extraction</td>
      <td>SImple xgb model</td>
      <td>CNN based model</td>
      <td>RNN based model</td>
    </tr>
  </tbody>
</table>


# Task 3

Current stage: Reading
=========================

Main schedual:
==============
<table>
  <thead>
    <tr>
      <th>Stage 1</th>
      <th>Stage 2</th>
      <th>Stage 3</th>
      <th>Stage 4</th>
      <th>Stage 5</th>
      <th>Stage 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Visualization & Statistics</td>
      <td>Data preprocessing: abnormal data / noise (auto encoder/fft)</td>
      <td>Slicing & statistics</td>
      <td>Feature extraction(Manural, auto-encoder, fft)</td>
      <td>Aggregation model</td>
      <td>Sequence model</td>
    </tr>
  </tbody>
</table>

# Task 2

Current stage: Reading
===================================

Main schedual:
==============
<table>
  <thead>
    <tr>
      <th>Stage 1</th>
      <th>Stage 2</th>
      <th>Stage 3</th>
      <th>Stage 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Code reconstructing</td>
      <td>Deal with imbalanced-data</td>
      <td>Feature selection</td>
      <td>Model selection and hyper-param tuning</td>
    </tr>
  </tbody>
</table>

References:
=================
https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/discussion/19240#110095
https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/discussion/20247#latest-356655
https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/discussion/20258#latest-133476
https://www.kaggle.com/c/telstra-recruiting-network/discussion/19239#latest-381687
https://www.kaggle.com/c/prudential-life-insurance-assessment/discussion/19003#latest-229720
https://www.kaggle.com/c/otto-group-product-classification-challenge/discussion/14335#latest-622005
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/discussion/18918#latest-627461
https://www.kaggle.com/c/mlsp-2014-mri/discussion/9854#latest-568751
https://github.com/diefimov/santander_2016/blob/master/README.pdf




# Task 1

Current stage: Hyper-params tuning
===================================

Best result:0.7389
=================

Best Params:
=================
<table>
  <thead>
    <tr>
      <th>Filling Method</th>
      <th>pre-feature number</th>
      <th>feature selection model(2nd)</th>
      <th>Model</th>
      <th>Outlier Detection</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random forest</td>
      <td>200</td>
      <td>126 (lasso selection with alpha 0.02)</td>
      <td>Ensemble</td>
      <td>Ask Chen Le</td>
    </tr>
  </tbody>
</table>

Main schedual:
==============
<table>
  <thead>
    <tr>
      <th>Stage 1</th>
      <th>Stage 2</th>
      <th>Stage 3</th>
      <th>Stage 4</th>
      <th>Stage 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Code reconstructing</td>
      <td>Fill the missing data</td>
      <td>Outlier detection</td>
      <td>Feature selection</td>
      <td>Model selection and hyper-param tuning</td>
    </tr>
  </tbody>
</table>

Ensemble Reference:
https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
