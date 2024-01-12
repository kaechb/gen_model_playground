Features
========

* Template code for various generative models such as GANs, VAEs, etc.
* Easy-to-modify structures for quick customization and experimentation.
* Ideal for educational purposes and prototyping new generative model ideas.

Usage
=====

``gen_model_toys`` serves as a template for creating and modifying generative models. It contains GANs, VAE, Discrete NF, Flow Matching, Continuous NF, DDPM.
Check the ``__main__.py`` file for the possible argument configurations.
So in the end run ``python -m gen_model_playground --{args}`` to run the code, e.g. ``python -m gen_model_playground``.
Please refer to the documentation within each model template for specific modification tips and best practices.
It is recommended to create a Wandb account and use it for logging and visualization, as it is very easy to use and provides a lot of useful features.
A documentation for the project is available at https://gen-model-playground.readthedocs.io/en/latest/.
Contributing
============

If you have suggestions for additional templates or improvements, contributions are welcome!

License
=======

This project is licensed under the MIT License. See the `LICENSE <LICENSE>`_ file for more details.
