# cookiecutter-ml-course

A standardized template for doing and sharing data science work

## Getting Started

Follow these steps to set up your project environment:

### 1. Use This Template

Click the green **"Use this template"** button at the top of this repository, then select **"Create a new repository"**. Give your project a meaningful name.

### 2. Clone Your Repository

```bash
git clone https://github.com/your-username/your-project-name.git
cd your-project-name
```

### 3. Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv #python>=3.8+

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

### 5. Start Working!

You're now ready to start your data science project. Here are some common next steps:

- Add your raw data to `data/raw/`
- Create exploratory notebooks in `notebooks/`
- Develop data processing scripts in `src/data/`
- Build models in `src/models/`

## Project Organization

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
|   |
|    ├── taxi_ride      <- Source code dir in this project.
|        ├── data <- Scripts to download or generate data
│            └── preprocess_data.py
|        |
│        ├── features
│            └── build_features.py <- Scripts to turn raw data into features for modeling
|        |
│        ├── models
│            ├── train_model.py <- Scripts to train models and then use trained models to make 
|            |                     predictions
│            └── predict_model.py
|        |
│        └── visualization    <- Scripts to create exploratory and results oriented visualizations
│            └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```

## Useful Commands

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Deactivate virtual environment
deactivate

# Update requirements.txt after installing new packages
pip freeze > requirements.txt

# Run tests (if configured)
make test
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>