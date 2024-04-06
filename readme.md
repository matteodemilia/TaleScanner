# Tale Scanner
By Christopher Mead Matteo DeMilia Tiera Smith Kyrstn Hall 

Tale Scanner is a web application that can analyze text consisting 200-600 words. Built using `HTML/CSS/JS`, `Python Flask`, and Python libraries `spaCy` and `Gramformer`

## Current Features
to be added!

## Installation 

**Note:** Run using pip/python or pip3/python3 depending on Python version

1. Install [Python](https://www.python.org/downloads/)

2. Install [spaCy](https://spacy.io/usage):
  ```        
  pip install -U pip setuptools wheel
  pip install -U spacy
  python -m spacy download en_core_web_trf
  python -m spacy download en_core_web_sm
  ```
3. Install Flask, the web server
 ```
 pip install flask
 ``` 

4. Install [Gramformer](https://github.com/PrithivirajDamodaran/Gramformer)
```
pip install -U git+https://github.com/PrithivirajDamodaran/Gramformer.git
```

5. Install Torch - used by Gramformer
```
pip install torch
```

## Running

1. Compile and run the program
```
python analysis.py
```

 2. View website using local lost `127.0.0.1:5000 `
