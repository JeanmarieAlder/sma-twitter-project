# Tweet Analysis

Work in process. For a master lecture project at University of Fribourg, Switzerland.

## Introduction
This project aims to analyze a collection of tweets stored in a CSV file. The goal is to extract relevant information, filter out irrelevant tweets, and create a graph representation based on the similarities between tweets.

## Getting started


To get started with the Tweet Analysis, follow these steps:

1. Create a virtual environment named `venv` in the project directory. Adapth the command ```python``` if needed (python3, py):
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment. On Windows, you can activate it with:
   ```bash
   venv\Scripts\activate
   ```
   On Unix or MacOS, use:
   ```bash
   source venv/bin/activate
   ```

3. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. Once the dependencies are installed, you can run the program by executing the `main.py` script:
   ```bash
   python main.py
   ```

This will start the Tweet Analysis process, performing data preprocessing, graph generation, persistence, community detection, and result analysis as described in the main function.

## Experiments
Unit experiments are available in the experiments folder. Learn more about these scripts by reading the README.md file in the experiments folder.

## Contributors
- Anu and JM

## Sample Output
![Graph Visualization](experiments/graph.jpg)