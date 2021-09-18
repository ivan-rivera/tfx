# Tensorflow Extended Demo

In this repository I'm taking TFX for a spin.

## Requirements

Aside from `requirements.txt`, this project requires Python 3.8.12. Lastly, the original content of the `fraud` directory was populated with the following command:

```
 tfx template copy \
 --model=taxi \
 --pipeline_name="fraud" \
 --destination_path="fraud"
```

## Data

Since the dataset is fairly large, it is not committed to git, but it can be downloaded from [here](https://www.kaggle.com/mishra5001/credit-card?select=application_data.csv) and saved in `data/`. We are using a credit card fraud detection dataset.

## TFX resources

Here is a list of relevant TFX resources that were used for this exercise:

* [Building TFX Pipeline Locally](https://www.tensorflow.org/tfx/guide/build_local_pipeline)
* [Create a TFX pipeline using templates with local orchestrator - Colab Notebook](https://colab.research.google.com/github/tensorflow/tfx/blob/master/docs/tutorials/tfx/template_local.ipynb)

## How to get it working
* clone
* download `application_data.csv` from Kaggle to the top-level `data/` directory
* run `make sample_data`
* run `make create_pipeline`
* run `make update_and_run`
* run `make tensorboard` to check out the training logs
* Check out the notebooks (work in progress)

# TODO:
* Simplify
* Work out where run_args are getting set
* Investigate the error `AttributeError: 'NoneType' object has no attribute '__wrapped__'`
* Investigate missing features stats in the notebooks
* Look into the train/validation split
* Set `preprocessing_fn` in a notebook for exploration
* Either print or output model diagram
* Find a way to visualise the pipeline (kubeflow style)
* Add feature weight distributions into the tensorboard (+ play around with TF board)
* Add a resolver to fetch the most recent blessed model to be used in the evaluation
* Get tests working
* Get more familiar with Beam (and TFT Beam)
* Add some custom pre-processing logic
