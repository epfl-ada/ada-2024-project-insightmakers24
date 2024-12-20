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
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
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

We have examined the overall ratings for all beer to see whether people rate differently through time. We will also investigate whether there are certain moments in time where the ratings are higher or lower than on average to see if there are some trends for this beer that influences its rating. Lastly we will check if certain days or weeks have a notable difference and try to link these to holidays, events or festivals to see if these can influence the rating of the beer.

To do this we principally use the rating.csv file from both dataset. We have analyzed the overall evolution of the rating through the year for both dataset and use different metrics to see how the user’s rating evolves. Then, we will compute the average rating over time and check if we find significant disparities. This analysis will be conducted twice, first by month, to identify any seasonal effects or periodic trends, and second by day, to detect any patterns related to specific events and holidays. 

This will show us if there is a bias introduced in the rating from all these factors and how much they influence the ratings.

### Part 2: Impact of initial and Anchoring Effect 

As we all know, the perspective of others can influence our own opinion. So we can imagine that the initial rating will have a large impact on the final rating.
To see this influence, we compared the first rating with the mean of every other rating. We check the correlation between these ratings, and it appears to be higher than the correlation between the last rating and the others.
We performed Pearson and Spearman tests to see whether a higher initial rating correlates with a higher final rating. Both tests show us a clear correlation, which is expected since the rating concerne the same beer. We know that the effect of the first rating exists, it is known as  the anchoring effect, but for now we cannot affirm that what we observed was uniquely related to this effect. 
It would be to investigate further about the anchoring effect, and also check if the ratings that were the most recents at the time when someone gave a rating have an influence, or perhaps if the rating that content some text review enhances this effect.

### Part 3: Country biases investigation

For the initial analysis, we focus on the BeerAdvocate dataset, particularly: ratings.txt, users.csv and breweries.csv. Note that ratings.txt was converted to a csv for convenience. 
Histograms based on the ratings for both the ratings where the user comes from the same location (domestic) and the ratings where the user comes from another location (international), we notice a slight distribution difference. A t-test and mean comparisons suggest that users that come from the same country as the beer might indeed tend to give slightly higher ratings.

We also merged the beer consumption dataset to the ratings and the breweries to see if there is a relation between the mean ratings per location and the beer consumption per capita in this location. We find a significant correlation between the two variables, however the country bias partly explains this correlation so we will need to first correct the ratings from the country bias. We can also explore how the bias varies depending on the country or depending on the beer type.

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
- Oliver: Code and readme for the time analysis (part 1)
- Romain: Anchoring effect Investigation (part 2)
- Yann: Country biases analysis and readme (part 3)
- Edgar Desnos: Beer Names influence (part 4)