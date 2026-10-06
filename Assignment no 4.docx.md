# Text-only document extract

Source document: Assignment no 4.docx

Images and layout omitted. Claims below are source text, not independently verified results.

Assignment no 4:

Part I: Text Preprocessing

Standard preprocessing steps were applied to the tweet text data to clean and prepare it for model training. The steps included:

Converting text to lowercase

Removing URLs, mentions (@), and hashtags (#)

Removing punctuation and digits

Removing common English stopwords

Applying stemming using PorterStemmer

Example:

Original:@VirginAmerica it's really aggressive to blast obnoxious "entertainment" in your guests' faces. And they have little recourse

Cleaned:realli aggress blast obnoxi entertain guest face littl recours.



Part II: Classification Models and Results

The dataset was split into 70% training and 30% testing. The following models were trained and evaluated:

Decision Tree

Naïve Bayes

K-Nearest Neighbors (K=7)

Each model used a pipeline with TF-IDF vectorizer. Their performance was assessed using classification reports and accuracy.

Part III: Result Comparison:

Model

    Accuracy          

    Comment

Decision Tree

     0.6986

     May overfit but handles imbalance reasonably well.

Naïve Bayes

      0.6891

     Great for text, often best performing.

KNN (k=7)

     0.5303

      Slower, depends heavily on k.



Bonus part:



Virgin America appears to have relatively more positive tweets compared to others, even though overall tweet volume is lower than others.