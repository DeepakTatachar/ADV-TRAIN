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

.. automodule:: utils.normalize
   :members:

.. automodule:: utils.framework
   :members:

.. automodule:: utils.instantiate_model
   :members:

.. autoclass:: utils.framework.Framework
   :members: train, test, validate, __init__

.. autoclass:: utils.boundary_visualization_extension.VisualizeBoundaries
   :members: generate_decision_boundaries, __init__, show

.. autoclass:: utils.preprocess.preprocess
   :members: forward, back_approx

