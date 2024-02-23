To install spaCy, https://spacy.io/usage, does a lot of language processing:
                    
        pip install -U pip setuptools wheel
        pip install -U spacy
        python -m spacy download en_core_web_trf
        python -m spacy download en_core_web_sm

To install flask, the python web server:
        pip install flask

To install torch, used by gramformer:
        pip install torch

To install gramformer, https://github.com/PrithivirajDamodaran/Gramformer, used for verb errors:
        pip install -U git+https://github.com/PrithivirajDamodaran/Gramformer.git
