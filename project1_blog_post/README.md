# AirBnB Analysis of Seattle and Boston Data

## Running the project

### Libraries Used

* `pandas` (for data analysis)
* `matplotlib` (for plotting)
* `seaborn` (for plotting)
* `numpy` (for a little numerical work)
* `sklearn` (for Ridge Regression)

### Getting the data.

The data for the Seattle AirBnB listings can be found [here](https://www.kaggle.com/airbnb/seattle/data)
and the data for the Boston AirBnb listings can be found [here](https://www.kaggle.com/airbnb/boston).
In both cases, click on the `download` button that is next to the `New Notebook` button (there is also
another download link further down the page... don't use that one!).  Each one will download a zip file.
Move each zip file into `data/boston` and `data/seattle` directory, and unzip into their respective
`.csv` files.  The expected structure is shown further down.  Only the `.csv` files are important as far
as the proper execution of the notebook.

## Motivation for the project

AirBnb is a site that allows owners to list short-term rentals, and guests to rent them.  These are often
considered cheaper than hotels (whether this is actually true or not is its own interesting question,
but is not covered here), and often have more amenities, such as kitchens.

In investigating the data, I was curious about the following three questions:

1. What are the most important aspects of insuring a high review score?
2. What are the most important predictors of a more expensive listing?
3. What are the most expensive times of the year to rent on AirBnb?  Does this differ based on whether the
listing is in Seattle or Boston?
   
This portion of the repository contains a jupyter notebook that looks into these questions.

## Files in the repository.

Note that some of these files are not part of the repo, but are instead created by the notebook or must
be downloaded for the notebook to function.  These are noted as such below (don't confuse those
parenthetical remarks as being part of the file name!)

The main work is done in `Udacity\ Making\ A\ Data\ Science\ Blog\ Post\ Project.ipynb`, which reads
from the `.csv` files in the `data` directory (which in the gitignore and thus not actually checked
into the repo).

```
udacity-data-science
├── README.md
├── main.py
└── project1_blog_post
    ├── README.md
    ├── Udacity\ Making\ A\ Data\ Science\ Blog\ Post\ Project.ipynb
    ├── boston.png (created by notebook)
    ├── checkin_review.png (created by notebook)
    ├── cleanliness_review.png (created by notebook)
    ├── communication_review.png (created by notebook)
    ├── data (must be downloaded.  The .csv files are what are required in the end)
    │        ├── archive.zip
    │        ├── boston
    │        │       ├── archive\ (1).zip
    │        │       ├── calendar.csv
    │        │       ├── listings.csv
    │        │       └── reviews.csv
    │        └── seattle
    │            ├── calendar.csv
    │            ├── listings.csv
    │            └── reviews.csv
    ├── price.png (created by notebook)
    ├── property_review.png (created by notebook)
    ├── seattle.png (created by notebook)
    ├── vaccuracy_review.png (created by notebook)
    └── value_review.png (created by notebook)
```

## Results Summary

1. What are the most important aspects of insuring a high review score?
   * I found here that the type of property was the most important.  Not disclosing a property type at all was damaging,
        guesthouses and villas tended to score well.
2. What are the most important predictors of a more expensive listing?
   * Again, property type was the most important predictor.  Boats and guesthouses were the most expensive, tents and
     dorms were the cheapest.
3. What are the most expensive times of the year to rent on AirBnb?  Does this differ based on whether the
listing is in Seattle or Boston?
   * The most expensive time to visit Boston are the early Fall months of September and October.  Seattle is most
    expensive in the Summer months: June through September.
     
## Acknowledgements

* [Kaggle](http://www.kaggle.com) for providing the datasets in an easy to download format.
* [Inside AirBnB](http://insideairbnb.com/get-the-data.html) for creating the datasets in the first place
* [Udacity](http://www.udacity.com) for providing the instruction and inspiration for this project.