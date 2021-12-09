## To download:

`git clone https://github.com/NJindia/boxoffice_revenue_predictor`

## To create and activate virtual environment 
  1. Create Virtual Environment (Windows Only)  
  `cd ./boxoffice_revenue_predictor`  
  `pip install virtualenv`  
  `virtualenv venv`  
  `.\venv\Scripts\activate`
  2. Installing Dependencies  
  `pip install -r requirements.txt`
## To run data collection:

`python extract_data.py [options]`

### Available Options:

``--load_imdb_data``: loads IMDB datasets from ./data.*.tsv.gz. Only necessary if new data downloaded to replace old
tsv.gz files.

``--scrape_boxofficemojo``: scrapes BoxOfficeMojo.com with data obtained from loading imdb data. Can take up to 1-2 days
to complete. Only run if you have to update box office data or get new info for new movies added to dataset.

``--scrape_mpaa``: scrapes IMDB.com for missing MPAA ratings. Only necessary if information needed for new movies added
to dataset.

## To run model training and evaluation

`python models.py`

Note that any modifications to how models are trained have to be done in the `models.py` file, but the current
configuration is the one of the optimal. All grid search algorithms are commented out to save your CPU, uncomment at
your own risk :)