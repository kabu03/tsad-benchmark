# TSAD Benchmark Suite

**A unified framework for Time Series Anomaly Detection (TSAD) research.**

This project enables the systematic evaluation of anomaly detection algorithms on time-series data. It bridges the gap between classic statistical methods and modern deep learning architectures, offering a standardized pipeline to train, test, and rank detectors.

## Dataset

This benchmark utilizes the **Hexagon ML/UCR Time Series Anomaly Detection dataset**.

[Download the dataset here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip).

## Quickstart

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kabu03/tsad-benchmark
   cd tsad-benchmark
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the GUI:**

   ```bash
   python3 src/gui.py
   ```

   _Alternatively, use the CLI to run batch experiments (e.g., all deep learning models):_

   ```bash
   python3 -m src.run_all --group deep
   ```

## Models Included

The suite covers a wide spectrum of detectors:

| Category          | Models                                                                                       |
| ----------------- | -------------------------------------------------------------------------------------------- |
| **Traditional**   | • Isolation Forest<br>• Z-Score<br>• Local Outlier Factor (LOF)<br>• Matrix Profile Discords |
| **Deep Learning** | • Long Short-Term Memory (LSTM)<br>• Autoencoder<br>• Temporal Convolutional Network (TCN)   |

## Performance Metrics

Models are evaluated using rigorous ranking and classification metrics:

- **AUC-ROC** (Area Under the ROC Curve)
- **PR-AUC** (Area Under the Precision-Recall Curve)
- **Top-K Hit Rate** (Oracle Top-K accuracy)

## GUI Preview

![GUI Screenshot](assets/gui.png)

## Results & Visualization

- **Results:** Metrics are automatically saved as JSON files in the `results/` directory.
- **Visualization:** Use the `notebooks/02_visualizations.ipynb` notebook to analyze performance and generate plots.

## Contributions

Contributions are welcome! Please open an issue or pull request with a clear problem statement and reproducible steps.
