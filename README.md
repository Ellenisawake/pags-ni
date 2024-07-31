# The Northern Ireland Publicly Accessible Green Spaces (PAGS-NI) Dataset
Authors: [Jian Gao](https://pure.qub.ac.uk/en/persons/jian-gao-2), [Niall McLaughlin](https://pure.qub.ac.uk/en/persons/niall-mclaughlin), [Joanna Sara Valson](https://pure.qub.ac.uk/en/persons/joanna-sara-valson), [Neil Anderson](https://pure.qub.ac.uk/en/persons/neil-anderson), [Ruth Hunter](https://pure.qub.ac.uk/en/persons/ruth-hunter) 

<img width="1113" alt="system_overview_figure" src="https://github.com/user-attachments/assets/6e43c824-3bb2-4dfe-b649-0a44e3e5e842">
The study of the health effects of Publicly Accessible Green Spaces (PAGS), such as parks and urban greenways, has received increasing attention in environmental sciences and public health research. However, the lack of relevant data and methods for PAGS mapping limits this work. To our best knowledge, most of the existing studies of PAGS mapping are manual, limited to small regions, and do not generalise geographically.

In this paper, we introduce a first-of-its-kind dataset - the Northern Ireland Publicly Accessible Green Spaces (PAGS-NI) dataset. Unlike existing datasets that typically consider only visual remote sensing data, our PAGS-NI dataset combines high-resolution, multi-band remote sensing data, geographical information data and activity data with hand-verified PAGS ground truth. Using this dataset, we develop a semantic segmentation model for automatic and scalable PAGS mapping that fuses these different data sources. Our model is able to predict PAGS on unseen places given appropriate training, which exceeds prior art. Furthermore, we show that our model trained solely on Northern Ireland can generalise to PAGS prediction for areas in the United States. Our model and dataset have the potential to advance large-scale PAGS studies in environmental science and public health research.



# Paper
Coming soon.

# Dataset
Please send you request to j.gaoATqub.ac.uk for access.


# Code
Coming soon.
## Running model test (inference only)
Hardware requirement: a GPU with at least 2GB memory
Software environment:
CUDA 12.2
[PyTorch 2.2](https://pytorch.org/get-started/previous-versions/)
## Converting model output to GeoTIFF
[osgeo](https://pypi.org/project/osgeo/)

# How to cite
> @inproceedings{gao2024learning,
> author    = {Jian Gao and Niall McLaughlin and Joanna Sara Valson and Neil Anderson and Ruth Hunter},
> title     = {Learning to Segment Publicly Accessible Green Spaces with Visual and Semantic Data},
> booktitle = {BMVC},
> year      = {2024},
> }
