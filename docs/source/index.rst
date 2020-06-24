Welcome to ADV-TRAIN's documentation!
=====================================
This is a framework built on top of pytorch to make machine learning training and inference tasks easier. Along with that it also enables easy dataset and network instantiations, visualize boundaries and more.
This was created at the Nanoelectronics Research Laboratory at Purdue.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Modules
==================

.. automodule:: advtrain.utils.normalize
   :members:

.. automodule:: advtrain.framework
   :members:

.. automodule:: advtrain.instantiate_model
   :members:

.. autoclass:: advtrain.framework.Framework
   :members: train, test, validate, __init__

.. autoclass:: advtrain.boundary_visualization.VisualizeBoundaries
   :members: generate_decision_boundaries, __init__, show

.. autoclass:: advtrain.utils.preprocess.preprocess
   :members: forward, back_approx

.. automodule:: advtrain.attack_framework.multi_lib_attacks
   :members: