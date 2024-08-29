# Graph Language Models
This repository contains the code for the paper "[Graph Language Models](https://arxiv.org/abs/2401.07105)". 
Please feel free to send us an email (<a href="mailto:plenz@cl.uni-heidelberg.de">plenz@cl.uni-heidelberg.de</a>) if you have any questions, comments or feedback. 

<p align="center">
  <img src="./figs/GLM_overview.png" width="500" title="GLM" alt="Picture depicting the general concept of GLMs.">
</p>

## Huggingface
We have an implementation of the code on Huggingface which should be easy to include in your own projects: [https://huggingface.co/models?other=glm-t5](https://huggingface.co/models?other=glm-t5).

## Minimal working example
In `minimal_working_example.py` we provide a minimal working example to show how to load and use the classification models for inference. Note that these models are not trained, i.e., they are like the linear-probing setting from the paper. To use the minimal working example, you only need to install the requirements -- other steps are not necessary.


### 1 Requirements

To run is tested with python version 3.9.16 and the packages in `requirements.txt`. You can install the requirements by running:

```bash
pip install -r requirements.txt
```

Make sure that your `PYTHONPATH` includes this directory, for instance by including `.` and launching all codes from here. To achieve this, you can for example add the following line in your `.bashrc` file:

```bash
export PYTHONPATH=$PYTHONPATH:.
```


### 2 Data

You can download the ConceptNet data (94MB) by running:
```bash
wget https://www.cl.uni-heidelberg.de/~plenz/GLM/relation_subgraphs_random.tar.gz
tar -xvzf relation_subgraphs_random.tar.gz
mv relation_subgraphs_random data/knowledgegraph/conceptnet
rm relation_subgraphs_random.tar.gz
```

If you want to run the GNN baselines aswell you can download the data including the embeddings (45GB) by replacing the link above with `https://www.cl.uni-heidelberg.de/~plenz/GLM/relation_subgraphs_random_with_GNN_data.tar.gz`

To download the REBEL data (2.2GB) run:
```bash
wget https://www.cl.uni-heidelberg.de/~plenz/GLM/rebel_dataset.tar.gz
tar -xvzf rebel_dataset.tar.gz
mv rebel_dataset data/
rm rebel_dataset.tar.gz
```

Files used during preprocessing to compile the data are in `preprocessing`. However, currently it is difficult to provide download-links to the unprocessed data due to storage space limitations. We apologize for the inconvenience -- please send us an email if you need the data and we will provide it to you. 

### 3 Training and Evaluation
Python codes to train and evaluate the models are in `experiments`. In `scripts` there are example bash scripts to run the experiments. The parameters are explained in more detail in the python codes. 

To run the scripts, you can use one of the following commands:
```bash
bash scripts/conceptnet_relation_prediction/submit_LM.sh
bash scripts/conceptnet_relation_prediction/submit_GNN.sh
bash scripts/rebel_text_guided_relation_prediction/submit_LM.sh
bash scripts/rebel_text_guided_relation_prediction/submit_eval_LM.sh
```



## Citation
If you benefit from this code, please consider citing our paper:
```
@inproceedings{plenz-frank-2024-graph,
    title = "Graph Language Models",
    author = "Plenz, Moritz and Frank, Anette",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics",
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
}
```