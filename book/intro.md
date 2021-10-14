# CA4015 Clustering Assignment Introduction
---

## Introduction to the Iowa Gambling Task
This Jupyter Book will hold an analysis of data from 617 Healthy Participants Performing the **_Iowa Gambling Task_** (IGT). The data, which originates from 10 individual studies, is pooled together by {cite}`Steingroever2015`. All participants are healthy (have no known neurological impairments). Participants were assessed on a computerised version of the IGT. The playing conditions vary between each study, with participant's attempts ranging between *95 - 150 tries*. The *payoff scheme* varies also, with some servers hosting **harsher penalties** and **more lucrative rewards**.  

## Introduction to K-Means Clustering Algorithm
**_K-Means Clustering_** is a classical machine learning algorithm developed well over 50 years ago. K-Means is an *unsupervised* machine learning technique meaning, unlike classification it does not need to be trained on annotated training data. Although K-Means is a rhobust algorithm, it does suffer from the 'curse of dimensionality'. This means that techniques such as *Principal Component Analysis* are commonly used in conjunction with K-Means to perform dimensionality reduction.

## Introduction to Federated Learning
__*Federated Learning*__ (FL) is a deep learning approach which involves training a model over disconnected or siloed data centres such as mobile phones. Rather than both the data and model being centralised on one system, data is preserved in its local environment. A machine learning model is sent to the system hosting the data, rather than the reverse. This approach makes a step forward in protecting the privacy of user-generated data. In FL, the user data is not transmitted across a network. However, there are challenges associated with FL, including: *the expensive nature*, *system heterogeneity*, *statistical heterogeneity*, and *privacy* {cite}`Li2020`.

## Dataset Description 
As stated previously, the data originates from 10 individual studies. These studies can be found listed below:

| Study                     | Amount of Participants| Number of Trials  |
| :-------------:           |:-------------:		|     :-----:       |
| {cite}`FRIDBERG201028`    | 15			        |        95         |
| {cite}`hortsmanm2012`     | 162      		        |        100        |
| {cite}`KJOME2010299`      | 19      		        |        100        |
| {cite}`Maia16075`         | 40      		        |        100        |
| {cite}`PREMKUMAR20082002` | 25      		        |        100        |
| {cite}`stein_sub_study_1` | 70     		        |        100        |
| {cite}`stein_sub_study_2` | 57      		        |        150        |
| {cite}`WETZELS201014`     | 41      		        |        150        |
| {cite}`Wood2005`          | 153      		        |        100        |
| {cite}`Worthy2013`        | 35    		        |        100        |

### Quality Control
All studies were administered through a computerized version of the IGT to ensure quality {cite}`Steingroever2015`.