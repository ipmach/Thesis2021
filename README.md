# Leveraging Autoencoders for Robustness of Image Classifier based Applications

This code is from my Master of Artificial Intelligence thesis on Università della Svizzera Italiana (USI). It is a framework for large-scale experiments in the performance and behavior of different Autoencoders (AE) as supervisor systems with Out-Of-Distribution (OOD) and Ambiguous data. There are three main contributions:

- **Visualizer**: This allows us to visualize the behavior of different AEs.
- **Generation of Ambiguous data**: Our approach for generating ambiguous data.
- **Out-Of-Distribution Detection**:  Framework for testing the behavior of AEs with OOD data.

In this project, we implemented the next Autoencoders:
- Vanilla Autoencoder (Vanilla)
- Denoise Autoencoder (DAE)
- Sparse Autoencoder (SAE)
- Varational Autoencoder (VAE)
- Adversarial Autoencoder (AAE)
- Probabilistic Autoencoder (PAE)
- Ensemble Autoencoder - Randnet (EAE)

The code is divide in six different modules:
- **Visualizer**: Interfaces to visualize the AEs.
- **Compostela**: Ambiguous data generator method.
- **Databases Manager**: The module in charge of organizing and saving the data obtained in each experiment.
- **Autoencoders**: All the Autoencoders' implemenetations.
- **Datasets**: Use for the generation of the different datasets used in the project.

### Visualizer

Use to help visualize the latent space and OOD detection from the different Autoencoders.

- Visualize latent space <br />
For example, if we want to visualize the latent space of an Adversarial Autoencoder of latent space, dimension 2 executes:

``` bash
  ./visualize_AAE_2_space.sh
````
- Visualize latent space with interpolation tree <br />
For example if we want to visualize the latent space of an Adversarial Autoencoder with the interpolation tree of latent space dimension 2 execute:

``` bash
  ./visualize_AAE_2_space_tree.sh
````

- Visualize ood detection <br />
To visualize ood detection with Variational Autoencoders execute:

``` bash
  ./visualize_VAE_validator.sh
````

To visualize latent space, we call the internal function visualize_model.py. To visualize OOD GUI we call visualize_validation.py.

### DataSets

Use to create the datasets to train the different models. It has two internal scripts:

- no_filter.py

Generate a dataset and optionally OOD datasets to train the model with any anomaly applied. For example, if we want to generate a dataset of MNIST with MNIST-C and FashionMnist as outliers in folder_path:

``` bash
  src/python_code/no_filter.py folder_path "MNIST" "MNIST-C" "FashionMnist"
````

Important detail MNIST-C tries to load from a file the dataset. If the file does not exist, it will generate it.

- filter.py

Apply filter to the dataset, doesn't generate OODs. The permutation options allow us to create datasets for the training process used in the interpolation algorithm.

``` bash
  src/python_code/filter.py folder_path "MNIST" "Permutation" permutation_file value
````

The filterValues option allows us to filter specific values. For example, if we only want to have a MNIST with values 1 2 3, we do:

``` bash
  src/python_code/filter.py folder_path "MNIST" "FilterValues" 1 2 3
````

### Compostela

Our ambiguous data generator algorithm:

``` bash 
  ./train_script.sh
```

To execute only the ambiguous data generator without training new Autoencoders:

``` bash 
  python main_cmp.py
```


The non-official name come from the shape of our distribution data in the Latent Space, where the Discriminator value of our two label distributions are mountains, the Ambiguous space the valley where we are, and the tree we form a constellation in the sky. Thus, in Latin, "Campus Stellae" or "Compostela" in old Galician, a valley surrounded by stars. 

### Autoencoders

Every Autoencoder is a Keras Model and needs to have the next functions to operate the model:

``` python
  def encode(self, x): pass 
  def decode(self, z): pass
  def call_(self, inputs, training=None, mask=None): pass
  def load_weights_model(self, list_path): pass
  def save_weights_model(self, list_path): pass
````
And this ones to preprocess and post-process the data:

``` python
  def preprocessing(self, x): pass 
  def postprocessing(self, z): pass
  def preformat(self, x): pass 
  def postformat(self, z): pass
````

To execute an Autoencoder, you need to build it first (same method as Keras). The Autoencoder can be used in two ways to apply the reconstruction.
- Using the decode and encode functions:
``` python
  x_ = model.decode(model.encode(x))
````
- Or using all at onces:
``` python
  x_ = model(x)
````

A model can be trained with different Train classes. Each training class is a compilation and fit method at the same time. All Train classes have to inherit from Train Interface:

``` python
  def init_train(self, num_epochs, batch_size,
             learning_rate, db): pass
  def train(self, x_train, model_save, model_name, save_image=20,
            ood_data=None, ood_op=None, x_test=None, y_train=None): pass
  def init_dataset(self, x_train, y_train): pass
  def pre_model(self, x): pass
  def post_model(self, x): pass
  def do_img(self, x_train): pass
  def do_before_train(self, x_train, y_train): pass
  def do_after_train(self, x_train, y_train): pass
  
  @abstractmethod
  def do_save(self, model_save, model_name): pass
  @abstractmethod
  def do_epoch(self, dataset): pass
````
![Graph](https://github.com/ipmach/Interpolation/blob/main/Documentation/AE_class.png)

### Docker Compatibility

The different executions where design to be run in Docker Containers.

- To execute OOD AE experiments in parallel (2 or more GPUs):
``` bash 
  ./deploy_train_jobs_ood.sh
```
- To deploy containers for Ambiguous generation data.
``` bash 
  ./deploy_train_jobs.sh
```

### General Structure

![Graph](https://github.com/ipmach/Interpolation/blob/main/Documentation/Modules.png)
