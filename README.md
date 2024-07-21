<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <h1>SQ-Net Repository</h1>
</head>
<body>

<p>This repository contains the implementation of SQ-Net, a novel temporal encoding for Quantum Spiking Neural Networks (QSNN). The code includes data loading, quantum feature encoding, and a hybrid neural network model using TensorFlow and Qiskit.</p>

<h2>Files and Directories</h2>

<h3><b>data/</b></h3>
<ul>
    <li><b>Mackey-Glass Time Series(taw17).xlsx</b>: Mackey-Glass time series data.</li>
    <li><b>daily-minimum-temperatures.csv</b>: Daily minimum temperatures data.</li>
    <li><b>yahoo_data.xlsx</b>: Yahoo data for analysis.</li>
    <li><b>Temporal Encoding (Time Series Data).ipynb</b>: Jupyter notebook demonstrating temporal encoding using time series data.</li>
    <li><b>Temporal Encoding (Trigonometric Functions).ipynb</b>: Jupyter notebook demonstrating temporal encoding using trigonometric functions.</li>
</ul>


<h2>Setup Instructions</h2>

<h3>Clone the repository:</h3>
<pre>
<code>
git clone https://github.com/Ria-K912/SQ-Net.git
cd SQ-Net
</code>
</pre>

<h3>Install the required dependencies:</h3>
<pre>
<code>
pip install -r requirements.txt
</code>
</pre>

<h2>Requirements</h2>

<p>The <b>requirements.txt</b> file includes the following dependencies:</p>
<pre>
<code>
numpy
pandas
matplotlib
scikit-learn
tensorflow
qiskit
openpyxl
</code>
</pre>

<p>Make sure to have these installed before running the notebooks and scripts.</p>

<h2>Usage</h2>

<p>After installing the dependencies, you can explore the provided Jupyter notebooks for temporal encoding and time series analysis:</p>
<ul>
    <li><b>Temporal Encoding (Time Series Data).ipynb</b>: Demonstrates temporal encoding using time series data.</li>
    <li><b>Temporal Encoding (Trigonometric Functions).ipynb</b>: Demonstrates temporal encoding using trigonometric functions.</li>
</ul>

<h2>License</h2>

<p>This project is licensed under the MIT License - see the LICENSE file for details.</p>

</body>
</html>
