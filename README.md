# The biases behind rating: Uncovering the hidden influences in beer ratings

## Quickstart

```bash
# clone project
git clone git@github.com:epfl-ada/ada-2024-project-insightmakers24.git
cd ada-2024-project-insightmakers24

# [OPTIONAL] create conda environment
conda create -n insightmakers24
conda activate insightmakers24

# install requirements
pip install -r pip_requirements.txt
```

### How to use the library

- `BeerConsumption.csv` file need to be in `data/BeerConsumption.csv`.

- All other data directories (`BeerAdvocate`, `RateBeer` and `matched_beer_data`) need to be in `data/`.

- All `.txt` files has to be converted to `.cvs` files.

- `data_utils.py` contain every function use to load the dataframe used in `result.ipynb`

- Run the `results.ipynb` to have get our result.


## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   │   ├── time_bias.py                    <- Analysis functions for part 1
│   │   ├── anchoring_bias.py               <- Analysis functions for part 2
│   │   ├── country_bias.py                 <- Analysis functions for part 3
│   │   └── name_bias.py                    <- Analysis functions for part 4
│   │
│   └── utils                           <- Utility directory
│       └── data_utils.py                   <- Helper functions
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

## Abstract

The main goal of every rating app is to show an objective rating for various objects, which in our case are beers. As each beer receives a lot of reviews from different users the mean rate of each beer should be quite representative. However every rate is given by a human, and humans are not objective creatures, they might be influenced by several biases. If the majority of users are subject to the same bias, then the final rate can be significantly impacted. In this project, we want to analyze various biases, such as trends, cultural bias or naming bias, in the beer reviews’ dataset and see how they influence the rating. Knowing how people are influenced can help us to adjust the rating in order to get more objective and accurate ratings.

## Research questions

Throughout this analysis, we will try to answer the following questions:

1. How does time influence the ratings? Are there trends that bias the ratings and can we link them to holidays or festivals?
2. What is the influence of the first (or first few) ratings on the final rating? Are future reviewers biased by these first ratings?
3. Do people rate beers from their country differently than the rest of the world? Do high-beer consumer countries rate beers more favorably?
4. Does the beer’s name influence the final rating of consumers?

## Proposed Additional Datasets

Beer Consumption by Country

Data provided by World Population Review, available at https://worldpopulationreview.com/country-rankings/beer-consumption-by-country

This dataset includes information on the total and per capita beer consumption for the years 2020 and 2021.
We will use this data to explore whether a country's overall beer consumption influences the ratings and reviews of beers, both from those countries and by reviewers from those countries.


## Methods

### Part 1: Influence of time on the ratings

In this first part we have looked at the influence of time and the bias it can create on the rating for the beer. We have examined the overall ratings and the number of reviews for all beer to see how it evolves through time for both dataset, we did the same for the rating per style of beer for a sample of them. We have looked how these rating are affected by time. Lastly we have compared for two holidays the rating to see if these can influence the rating of the beer.

To do this we principally use the rating.csv file from both dataset. We have analyzed the overall evolution of the rating through the year for both dataset and use different metrics to measure and see how the user’s rating evolves. We separated the date to year, month and day and then separate the day linked to the holidays from the others to do the comparison on them.

This shows us that there is a bias introduced in the rating from some of these factors and how much they influence the ratings.

### Part 2: Impact of initial and Anchoring Effect 

As we all know, the perspective of others can influence our own opinion, this is known as the anchoring effect. 
So we can imagine that the initial rating will have a large impact on the final rating.
To see this influence, we first observed the disctibution of the rating over time for every individual beer. 
We observed that the rating given tend to converge around 3 with time, so we plot the distance between the mean rating and 3 (the average rating) and see that this tendancy appears for most of the beers. 
We performed Pearson and Spearman tests to see whether a higher initial rating correlates with a higher final rating. Both tests show us a clear correlation, which is expected since the rating concerne the same beer. We know that the effect of the first rating exists, it is known as the anchoring effect, but for now we cannot affirm that what we observed was uniquely related to this effect. 
Finally, we used the bros score of BeerAdvocate, and compare it with the overall rating of two groups: the beers with a very high or a very low first rating. 
We found that with a high first rating have a higher overall rating that the bros score, and with a low first rating have a lower overall rating that the bros score. 
According to a statistical T-test, these differences that these differences are significant.

### Part 3: Country biases investigation

In this part, we try to find if users that come from the same location as the beer put higher ratings for the beer.
We do the analysis on both BeerAdvocate and RateBeer datasets and compare the results to see if there is indeed a bias
in the ratings. Note that txt files have been converted to csv files for convenience.  

We first do a linear regression analysis on a subset of the data to see if domestic ratings have a positive influence
on the ratings. Then we plot jointly the histogram for domestic ratings and international ratings to see if there is
a shift in their distributions. We also do a boxplot of the ratings and show the means with error bars. Then we perform
a propensity score matching to balance the datasets and redo the above steps. Finally, we do a t-test on the domestic and 
international ratings to see if there is a significant difference in the ratings. All these steps are done on both the 
BeerAdvocate and RateBeer datasets.  

We also merge the beer consumption dataset to the ratings and the breweries to see if there is a relation between the mean ratings per location and the beer consumption per capita in this location.

### Part 4: Influence of Beer Names on Ratings

To investigate whether the name of a beer impacts its final rating, we will use text analysis techniques on beer names. Specifically, we will: Extract keywords from beer names. Perform sentiment analysis on these names to see if certain types of names (e.g., "Premium," "Classic") correlate with higher or lower ratings. Use statistical tests (e.g., Chi-square test) to evaluate whether specific words in the name are associated with significant rating differences. This analysis could reveal if certain types of names set expectations that influence user ratings, either positively or negatively. So far, keyword analysis yielded good initial results. However, our first attempts at sentiment analysis fell short but we will try using other techniques and libraries in hopes of better results.



## Proposed timeline

- 15.11.2024 Data Handling and Preprocessing & Initial Exploratory Data Analysis
- 30.11.2024 Implementation and Preliminary Analysis
- 07.12.2024 Compile Final Analysis
- 14.12.2024 Data Story Writing
- 20.12.2024 Milestone 3 Deadline


## Organization within the team

- Huiyun Zhu: Writing up the report or the data story, preparing the final presentation
- Oliver: Code/discussion, readme and data story for the time analysis (part 1)
- Romain: Anchoring effect Investigation (part 2)
- Yann: Country biases analysis and readme and data story (part 3)
- Edgar Desnos: Beer Names influence (part 4)