# OxSpred
OxSpred is an XGboost-based model that annotates the intracellular oxidative stress state of cells using transcriptome profiling.

## Installation

The script was developed under `Ubuntu 20.04.3 LTS`.

### Dependencies

- Python 3
- Pandas
- Numpy
- Scikit-learn

### Install OxSpred

```bash
git clone https://github.com/PeterPYChen/OxSpred.git
```

### Install the required packages with Anaconda

```bash
cd OxSpred
conda env create -f env.yml -n OxSpred
conda activate OxSpred
```

## Running OxSpred

### Parameters

- `--i` Input normalized gene expression matrix
    - File format: The input file should be in .csv format
    - Normalization criteria: LogNormalized, scale factor=10000
    - Unfiltered gene expression matrix
    
    ```bash
    ,cell1,cell2,cell3 ...
    gene1,0.0,2.468191,0.0
    gene2,0.0,1.856382,4.549756
    gene3,1.029684,4.186718,0.0
    ...
    ```
    
- `--m` Output probability or binary class. (default: binary)
    - binary
    - prob
- `--o` Prefix for the prediction output file. (default: osi_prediction.csv)
- `--t` Threads to use. (default: 8)

### **Example of usage**

The essential input to OxSpred is a normalized expression matrix from a single-cell dataset. The user can acquire the requirement file via various single-cell analysis tools. The demo dataset was generated via Seurat.

```r
library(Seurat)

# Load the dataset
data <- Read10X(data.dir = "../data/dataset")
seurat_object <- CreateSeuratObject(counts = data, min.features = 200) 
```

```r
seurat_object <- NormalizeData(seurat_data , verbose = F, normalization.method = "LogNormalize", scale.factor = 10000)
assay = GetAssayData(object = seurat_object , assay = "RNA",slot="data")
write.csv(assay, "expression.csv")
```

Execute OxSpred with Python.

```bash
python OxSpred.py --i expression.csv --o output.csv --t 8 --m prob
```

Import the OxSpred output file into R and annotate the cellular oxidative levels to the corresponding single cell data object (in this case Seurat object).

```r
pred_ROS <- read.csv("output.csv")
pred_ROS$cell <- sub(".1$", "-1", pred_ROS$cell)
pred_ROS.names <- pred_ROS[,3]

pred_ROS.df <- subset(pred_ROS,select="X1")
names(pred_ROS.df)[1] <- "ROSscore"
pred_ROS.df$ros_label <- ""
pred_ROS.df$ros_label[which(pred_ROS.df$ROSscore >= 0.5)] = "ROS+"
pred_ROS.df$ros_label[which(pred_ROS.df$ROSscore < 0.5)] = "ROS-"
row.names(pred_ROS.df) <- t(pred_ROS["cell"])

seurat_object <- AddMetaData(
  object = seurat_object,
  metadata = pred_ROS.df["ros_label"],
  col.name = "ROS.label"
)
seurat_object <- AddMetaData(
  object = seurat_object,
  metadata = pred_ROS.df["ROSscore"],
  col.name = "ROS.score"
)
```
