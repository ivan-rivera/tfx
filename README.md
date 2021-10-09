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
* [TFX in interactive context](https://www.adaltas.com/en/2021/03/05/tfx-overview/)
* [TFX in a notebook](https://notebook.community/tensorflow/tfx/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_interactive)

## How to get it working
* clone
* download `application_data.csv` from Kaggle to the top-level `data/` directory
* create a new folder `pipeline_outputs` in the project folder
* run `make sample_data`
* run `make create_pipeline`
* run `make update_and_run`
* run `make tensorboard` to check out the training logs
* Check out the notebooks (work in progress)

Note that with every run you are accumulating output data.

## TODO:
* Add a custom component