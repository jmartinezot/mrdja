.. mrdja documentation master file, created by
   sphinx-quickstart on Thu Mar 30 22:59:28 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mrdja's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodapi:: geometry
	:skip: svd
.. automodapi:: drawing
.. automodapi:: sampling
	:skip: sample_point_alligned_parallelogram_2d, sample_point_circle_2d, sample_point_circle_3d_rejection
	:skip: sample_point_cuboid, sample_point_parallelogram_2d, sample_point_parallelogram_3d, sample_point_sphere
	:skip: sampling_circle_3d_rejection, sampling_cuboid, set_random_seed
.. automodapi:: ransac
.. automodapi:: coreransac
.. automodapi:: coreransaccuda
.. automodapi:: coreransacutils

.. automodapi:: procrustes
	:skip: BaseEstimator, ClassifierMixin

.. highlight:: python

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
