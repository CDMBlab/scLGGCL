# scLGGCL: Label-Guided Graph Contrastive Learning for Single-Cell Clustering
Single-cell RNA sequencing (scRNA-seq) captures gene expression at single-cell level to uncover cellular heterogeneity, with clustering being a key analysis task. Existing methods often fail to integrate cell attributes and intercellular relationships, and most graph contrastive learning approaches ignore node similarity. We propose scLGGCL, a label-guided graph contrastive learning method for single-cell fusion clustering. A dual-reconstruction module fuses attribute and structural information, a label-guided contrastive module captures semantic node similarity using predicted labels, and a deep embedding clustering module optimizes results by drawing cells toward cluster centers. Experiments on single and cross-datasets validate scLGGCL's effectiveness.

## run environment

python --- 3.8

pytorch --- 1.11.0

torchvision --- 0.12.0

torchaudio --- 0.11.0

scanpy --- 1.8.2

scipy --- 1.6.2

numpy --- 1.22.3

leidenalg --- 0.10.2

## file Description

["model.py"]: Containing the complete model framework of scLGGCL, including the dual-reconstruction information fusion module and the label-guided graph contrastive learning module.

["loss.py"]: Defining multi-task loss functions: reconstruction loss, contrastive loss, and clustering loss.

["clustering.py"]: Implementing the deep embedding clustering self-optimization module, including cluster center initialization and target distribution alignment.

["train.py"]: Containing the main pipeline for model training and clustering execution.

["utils.py"]: Containing data preprocessing, cell-cell graph construction, evaluation metrics calculation, and visualization utilities.

## Data Source
The preprocessing of scRNA-seq datasets is provided by the utility functions in [`utils.py`]. Processed datasets in `.h5ad` format should be placed in the [`Data/`] directory, which is the default path for model input. If you want to analyze your own scRNA-seq datasets, please copy your `.h5ad` files to this directory and specify the file name using the `--data_path` argument.

## Usage
For applying scLGGCL, the convenient way is to run [`train.py`]. Please place the scRNA-seq dataset you want to analyze in the directory [`./Data/`], which is the default path for model input.

If you want to evaluate the similarity between the predicted clustering results and the true cell labels (based on NMI or ARI score), please provide your true labels in the `adata.obs['celltype']` field and set the argument `--celltype_key` to **'celltype'** and `--eval` to **True**.


## Arguments

    "--lr": default: 0.001. Description: Learning rate. default: 1e-3.

    "--beta": default: aa[i]. Description: Beta for CE loss. default: varies with dataset.

    "--gamma": default: bb[i]. Description: Gamma for contrastive loss. default: varies with dataset.

    "--delta": default: cc[i]. Description: Delta for graph reconstruction loss. default: varies with dataset.

    "--lambda_": default: dd[i]. Description: Lambda for DEC loss. default: varies with dataset.

    "--phi": default: 0.5. Description: phi, balance parameter for positive/negative samples. default: 0.5.

    "--dims": default: [1000, 256, 64, 32, 256]. Description: The number of neurons in each layer of the encoder. Advised to adjust based on dataset size.

    "--training_epoch": default: 100. Description: Number of epochs for pre-training stage. default: 100.

    "--clustering_epoch": default: 100. Description: Number of epochs for clustering stage. default: 100.

    "--device": default: "GPU". Description: Use GPU for training, or set to "False" to use CPU.

    "--sample_size": default: 1.0. Description: Dropout rate (1 - keep probability). Range: 0 to 1.

    "--debias": default: 0.0. Description: Debias factor for contrastive learning. default: 0.

    "--clustering": default: 'Kmeans'. Description: Clustering algorithm to use. Choices: ['Kmeans', 'PIC'].

    "--verbose": default: False. Description: If set, print detailed training information.

    "--data_aug": default: 1. Description: Whether to perform data augmentation (1: yes, 0: no).

    "--neg_type": default: 0. Description: Negative sample selection strategy. 0: selection, 1: no selection.

    "--pos_type": default: 0. Description: Positive sample selection strategy. 0: selection, 1: no selection.

other arguments:

    "--celltype_key": default: "celltype". Description: The column name in adata.obs containing true cell labels for model evaluation.

    "--save_pred_label": default: False. Description: To choose whether saves the predicted labels to the directory "./pred_label".

    "--save_model_para": default: False. Description: To choose whether saves the model parameters to the directory "./model_save".

    "--save_embedding": default: True. Description: To choose whether saves the cell embeddings to the directory "./embedding".

    "--max_num_cell": default: 4000. Description: Conduct random sampling training on large datasets. 4,000 is the maximum cells that a GPU (8GB RAM) can handle. In the experiment, scLGGCL still performs well when 1/10 cells is sampled for model training.


Dataset Download
The dataset is available for download via the following Google Drive link:https://drive.google.com/file/d/1oUT2a5rcVa3uVYazpf7WVqcekwqALRos/view?usp=drive_link

