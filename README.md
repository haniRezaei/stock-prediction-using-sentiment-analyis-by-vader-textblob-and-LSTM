# stock-prediction-using-sentiment-analyis-by-vader-textblob-and-LSTM
This project is a hybrid financial prediction system that combines sentiment analysis of news headlines with stock market data to model the behavior of the Dow Jones Industrial Average (DJIA). It is structured into two main parts: 
* Part 1: Classification – Predicting Stock Market Direction (to predict it will go Up/Down)
* Part 2: Regression – Predicting Actual DJIA Price (Adj Close)
in this project the Sentiment analysis of financial news articles is done using two different algorithms to calculate sentiment scores from each algorithm. The algorithms used include VADER from NLTK which give scores like positive, negative, neutral, and compound. VADER checks Polarity, the intensity of the emotion by checking how intensely the statement is positive, negative, or neutral. the second one is TextBlob from NLP scores the subjectivity and polarity of the financial articles. * TextBlob and VADER each capture different aspects of sentiment: TextBlob provides polarity and subjectivity metrics, while VADER offers fine-grained emotional strength through positive/neutral/negative/compound scores and handles negation and intensity. Combining both creates a richer sentiment representation.
* These two sentiment scoring sets are used in combination with each other. We combine the sentiment scores from the given two algorithms and the features from the historical stock price data.
  This is the final dataset that we want to apply machine learning. To check stock market movement prediction that uses news sentiment and stock price data to predict whether the Dow Jones Industrial Average        (DJIA) will go up or down the next day.

* in first part we focus on building machine learning models to classify whether the stock market will rise or fall on a given day, based on both textual sentiment from news and historical stock indicators.
  News headlines from multiple sources are aggregated and cleaned using natural language processing (tokenization, lemmatization, stopword removal).
  Multiple classification models such as Logistic Regression, Naive Bayes, SVM, KNN, Random Forest, XGBoost, and Linear Discriminant Analysis are trained and evaluated to predict the binary movement of the market   (0 = fall, 1 = rise or stay).
  The models are evaluated using metrics like accuracy, F1-score, and confusion matrix to compare their performance.
  this part shows how well machine learning algorithms can classify market direction using both quantitative and textual sentiment signals.

* the second part of the project aims to forecast the actual stock price (Adj Close) using a deep learning approach with LSTM (Long Short-Term Memory) networks, which are well-suited for time-series prediction.
  The same set of sentiment features (VADER, TextBlob) and stock market features are used.
  The data is scaled using MinMaxScaler and transformed into sequences of 10 time steps to feed into the LSTM.
  A multi-layer LSTM model with dropout layers is built to predict the next day’s DJIA adjusted closing price.
  The model is trained and validated using early stopping to prevent overfitting.
  Predictions are evaluated using regression metrics such as MSE, RMSE, MAE, R² Score, and MAPE.
  The predicted prices are visualized alongside the actual prices to show the model’s performance visually, along with the training loss curves.
  This part demonstrates how deep learning can capture time-based trends in financial and sentiment data to predict actual stock prices, not just movement direction.

