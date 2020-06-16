# Help For All (Mozilla Open Lab 2020)

    website created about donor and requestor listings under the Mozilla Spring Builders Open Lab Program, 2020.

# About Help For All

There is an ocean of information on the internet that can be put to use for the welfare of people. Especially when it comes to any global crisis, the Internet and social media stand out as a powerful tool which if utilized properly, works wonders. There’s a pool of resources and donations which do not always make way to the people who actually need them.

Help for All aims to match the donors to the appropriate requesters and vice versa, keeping the resource factor in mind so that people are able to access the donations they actually need and the donor resources are properly mapped. Help For all utilizes the power of online communities in facilitating the overall donation process and provides a solution to make the process more efficient by mapping requesters to donors and vice versa. We gather information from Twitter about tweets made on offers and requests pertaining to tangible donations of essential goods and make the information specific to user requirements readily available to the user. You just need to let us know whether you want to donate or request and mention the resource you want to make a donation or request for and we help you find your donor/offer match.

### What We Offer
  - ***For Donors***: Maximization of your donation impact by matching to the right. Go to our homepage, and let us know your offer in the search bar, for eg., type in “I want to donate clothes ” and we would return the tweets pertaining to the request of clothes, with the tweet location, date & time and the tweet link for you to access.

  - ***For Donation Seekers***: Find the right place that would help you in your crisis, request for what you need at the right place. Go to our homepage, and let us know what you want to request for, for eg., type in “I want to request groceries ” and we would return the tweets pertaining to the donations of groceries, with the tweet location, date & time for you to choose the appropriate tweet and also the tweet link for you to access.

## How it works
- Twitter is a great place to communicate with and learn about the recent events of interest. 
- We gather useful tweets from the Twitter API and classify them as donation/non donation types, and further perform matching for the request/offer and the donation resource types, for the ease of tracking tweets pertaining to a particular requirement. 
- This is facilitated through the predictions made by our machine learning models and the advanced NLP techniques employed.

### Core idea behind generating and classifying the data (usage of Machine Learning)

### Run the code
    - git clone [repository]
    - git push heroku master
    - heroku logs --tail
    - Go to url: https://help-for-all.herokuapp.com

## Technological Aspects 
- ***Languages:*** Python v3, HTML, CSS, Javascript
- ***Frameworks:*** Twitter API, pandas, nltk, re, numpy, collections, joblib, sklearn (MultinomialNB, TfidfVectorizer, pipeline, train_test_split, accuracy_score), Flask, Bootstrap, BeautifulSoap (bs4)
- ***Tools:*** Google Colaboratory, Heroku, GitHub

## Contents

    - template folder: HTML files for different web pages of the website
      (1) index.html
      (2) about.html
      (3) contact.html
      (4) trying.html
    - static folder: Contains styling and images for web pages
    - models: Contains the three models for classifying tweet into donation / non-donation, request/offer and resource type.
      

### URL: 
https://help-for-all.herokuapp.com



