# TSAD Benchmark Suite

This project evaluates a mix of classic and deep learning detectors using ranking-based metrics (AUC-ROC, PR-AUC, Top-K Hit Rate). It provides both a CLI pipeline and a GUI for easy configuration and execution.

## GUI Preview

![GUI Screenshot](assets/gui.png)

## Dataset

This benchmark uses the **Hexagon ML/UCR Time Series Anomaly Detection dataset**.
[Download the dataset here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip).

## Models

The benchmark includes a variety of traditional and deep learning models:

| Category          | Models                                                                                       |
| ----------------- | -------------------------------------------------------------------------------------------- |
| **Traditional**   | • Isolation Forest<br>• Z-Score<br>• Local Outlier Factor (LOF)<br>• Matrix Profile Discords |
| **Deep Learning** | • Long Short-Term Memory (LSTM)<br>• Autoencoder<br>• Temporal Convolutional Network (TCN)   |

## Metrics

Model performance was evaluated using:

- **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve)
- **PR-AUC** (Area Under the Precision-Recall Curve)
- **Top-K Hit Rate** (Oracle Top-K)

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

You can also run specific model groups via CLI, for example, to run all deep learning models on all datasets:
   ```bash
   python3 -m src.run_all --group deep
   ```

## Results & Visualization

- **Results:** Saved as JSON files in the `results/` directory.
- **Visualization:** Analyze metrics using the logic in the `notebooks/02_visualizations.ipynb` notebook.

## Contributions

Contributions are welcome! Please open an issue or submit a Pull Request with:

- A clear problem statement.
- Reproducible steps.
- Measured impact.
