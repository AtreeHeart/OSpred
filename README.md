# OSpred

An XGboost model for predicting intracellular oxidative stress status in the mouse spinal cord.

## Installation

The script was developed under `Ubuntu 20.04.3 LTS`.

### **Dependecies**

- Python 3
- Pandas
- Numpy
- Scikit-learn

### Install OSpred using git clone

```bash
git clone https://github.com/PeterPYChen/OSpred.git
```

### Install the required packages using Anaconda

```bash
cd OSpred
conda env create -f env.yml -n OSpred
conda activate OSpred
```

## Runngin OSpred

### Parameters

- `--i` Input the expresion matrix (LogNormalized, scale factor=10000) in csv format.

    ```bash
    			gene1,gene2,gene3 ...
    cell1,0.0,2.468191,0.0
    cell2,0.0,1.856382,4.549756
    cell3,1.029684,4.186718,0.0
    ...
    ```

- `--m` output probability or binary class. (default: binary)
    - binary
    - prob
- `--o` prefix for the prediction output file. (default: osi_prediction.csv)
- `--t` threads to use. (default: 8)

### **Example of usage**

```bash
python OSpred.py --i expression.csv --o output.csv --t 8 --m binary
```
