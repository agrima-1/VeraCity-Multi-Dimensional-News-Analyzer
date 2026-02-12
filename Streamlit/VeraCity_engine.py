import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup
from textblob import TextBlob
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

nlp = spacy.load('en_core_web_sm')

# --- 2. CLEANING FUNCTION ---

def clean_text_for_length(text):
    """
    Applies basic cleaning steps (HTML, URLs, Emojis, Whitespace) to the text column; which is treated as a Series here.
    Light cleaning for exploratory analysis and getting text length statistics.
    Does NOT perform any normalization of text like lemmatization, stopword removal, lowercase, remove punctuations.
    """

    # 1. Remove HTML tags - using the BeautifulSoup library and its built in HTML parsing engine
    
    def remove_html_tags(text):
        if pd.isna(text) or text is None:
            return text
         
        cleaned_html = BeautifulSoup(text, 'html.parser').get_text()

        # Fixes the merged words (e.g., "AnimalsTerrestrial" -> "Animals Terrestrial")
        cleaned_html = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned_html)

        return cleaned_html
    
    # 2. Remove URLs - Using same regex pattern as in the count_url function above.
    url_pattern = r'https?://\S+|www\.\S+|\S+\.\S+/\S+'

    # 3. Remove Emojis / symbols
    emoji_pattern = re.compile(
        '['
        '\U0001F600-\U0001F64F'  # emoticons
        '\U0001F300-\U0001F5FF'  # symbols & pictographs
        '\U0001F680-\U0001F6FF'  # transport & map symbols
        '\U0001F700-\U0001F77F'  # alchemical symbols
        '\U0001F900-\U0001F9FF'  # supplemental symbols and pictographs
        ']+', 
        flags = re.UNICODE
    )

    # Applying cleaning sequentially
    cleaned_text = text.astype(str).apply(remove_html_tags)
    
    cleaned_text = cleaned_text.str.replace(url_pattern, ' ', regex = True)
    
    cleaned_text = cleaned_text.str.replace(emoji_pattern, ' ', regex = True)

    # 4. Normalize Whitespace
    cleaned_text = cleaned_text.str.strip()
    cleaned_text = cleaned_text.str.replace(r'\s+', ' ', regex=True)

    return cleaned_text


def get_bias(text):
    """
    INPUT: text (row-wise) from usable_data DF
    OUTPUT: bias of text
    """
    text = str(text)
    text = TextBlob(text)
    bias = text.sentiment.subjectivity
    return bias

def get_intensity(text):
    """
    INPUT: text (row-wise) from usable_data DF
    OUTPUT: polarity of text (how negative or positive the tone is)
    """
    text = str(text)
    text = TextBlob(text)
    polarity = text.sentiment.polarity
    return abs(polarity)            
    # we take the absolute value because extremely positive (1.0) and extremely negative (-1.0) both are high intensity


def nlp_pipeline_optimized(texts_series):
    """
    INPUT: pandas series; clean_text
    
    Purpose of this function:
    - Lowercases everything
    - Tokenizes (split words)
    - Lemmatizes (finding word roots)
    - Removes Stopwords (common, but not useful words: the, is, at)
    - Removes Punctuation (but KEEPS numbers; numbers are often significant in news texts)

    OUTPUT: list of cleaned strings
    """

    processed_texts = []

    # Adding a limit of 5000 words.
    texts_to_process = texts_series.astype(str).apply(lambda x: " ".join(x.split()[:5000])).str.lower()

    # nlp.pipe takes an ITERABLE (the whole column)
    # batch_size = 100 means it processes 100 articles (100 rows) at a time in memory
    
    docs = nlp.pipe(texts_to_process, batch_size = 100)                                # Creates a doc generator; use nlp.pipe for batch processing, optimization, and to speed up
                                                                                       # nlp.pipe() can take a pandas Series as input because a Series is an iterable of strings

    
    for doc in docs:                                                                 # Creating the doc object from generator for each row at a time(this tokenizes the text and stores the tokens)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
        
        # newlines and tabs are treated as spaces, so we must explicitly use 'not token.is_space' eventhough spacy does not generally consider spaces as separaate tokens.
        # Extracts root of the word (the lemma) IF it's not a stopword or punctuation or a white space
        
        processed_texts.append(" ".join(tokens))
        
    return processed_texts

# NOTE: a doc object is created for every row separately still, but now 100 rows are processed parallely.
# so saying 'for doc in docs' still gives us one article at a time.

def clean_for_humans(text):
    # At this stage, I want to keep punctuation and capitalization, for the bias and intensity calculations
    # Using the clean_text_for_length function used on the FineFake dataset above
    # Removes HTML, URLs, and Emojis but KEEPS casing and punctuation;
    # Also spaces out two capitalized words with no space (E.G.: AnimalsTerrestrial)
    text_series = pd.Series([text])
    cleaned = clean_text_for_length(text_series)
    return cleaned[0]

def clean_for_vectorization(text):
    # Running the full nlp_pipeline_optimized
    # Lemmatizes and removes stopwords so TF-IDF can read it
    text_series = pd.Series([text])
    nlp_text_list = nlp_pipeline_optimized(text_series)
    return nlp_text_list[0]

def news_analysis_engine(user_input, bundle):
    
    # FAKE-LIKELIHOOD SCORE
    
    # STEP 1: Basic cleaning (HTML/URLs) - this version is for tone / bias / LLM
    human_readable_text = clean_for_humans(user_input)
    
    # STEP 2: Deep NLP cleaning- this version is ONLY for the ML Model; the version which will be vectorized using tf-idf
    vectorization_version = clean_for_vectorization(human_readable_text)
    
    # --------------------------------------------------------------------------------------------------------------------------------------------
    # STEP 3: Fake News Likelihood
    vectorized_data = bundle['vectorizer'].transform([vectorization_version])
    prediction_prob = bundle['model'].predict_proba(vectorized_data)        # Returns a 2D numpy array (n_samples, n_classes)
    fake_likelihood_score = prediction_prob[0][0] * 100              # Selects row 0 (the first sample) and column 0 (probability of class 0 = Fake)
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    
    # BIAS SCORE
    
    # STEP 4: Bias Score (Using Human Version)
    blob = TextBlob(human_readable_text)
    current_subjectivity = blob.sentiment.subjectivity
    
    # STEP 5: Calculating the Z-Score (The distance from mean of real news; NOTE: the mean and std from the FineFake REAL (1) news (training + testing) 
    # is used to serve as a benchmark; to keep comparisons fair
    
    bias_diff = current_subjectivity - bundle['REAL_BIAS_MEAN']        # REAL_BIAS_MEAN = mean of bias of ALL real-labeled texts in filtered dataset
    bias_z_score = bias_diff / bundle['REAL_BIAS_STD']                 # bias_z_score = how many standard deviations away is text’s subjectivity from real‑news subjectivity.


    
    # STEP 6: Converting the bias_z_score to a percentage (scale 0 - 100)
    
    # NOTE: 1.25 STD away (either in positive or negative direction) is considered to be 50% biased using 40 as the multiplier. 
    # So 2.5 STDs away from the mean is 100% biased.
    # By using a multiplier of 40:
    # 1.0 STD away = 40% Bias Score
    # 2.0 STD away = 80% Bias Score
    # 2.5 STD away = 100% Bias Score (Top 1% of outliers)
    # This is inspired from the empirical rule of standard deviation, which states that roughly 68% of data is WITHIN the range of 1 STD from the mean.
    # So if an article is 1 STD away from the mean here (NOT WITHIN 1 STD, but >= 1 STD), it is moving away from the 'most common' zone, where most data lies.
    # The rule also states that 95% of all data should fall within 2 STDs from the mean.
    # So if a text is exactly 2.5 (or more) STDs away from the mean [I.E., further away than 95% of the texts from the mean], it should get a bias score of 100%.
    # So 1.25 STD = 50% bias score and 2.5 STDs = 100% bias score; this is the conversion method used to convert from Z-score to a bias scale 0-100
    
    
    final_scaled_bias_score = abs(bias_z_score) * 40
    if final_scaled_bias_score > 100:
        final_scaled_bias_score = 100
    # In case score exceeds 100 when converting to a scale of (0-100), consider max always: 100
    # this is the user-display scaled bias score
    

    # -------------------------------------------------------------------------------------------------------------------------------------------------

    # POLARITY
    
    # STEP 7: Getting Polarity (Intensity) - use 'abs' because extreme +/- both show high emotional language
    # Intensity is the strength of the emotion
    current_polarity = blob.sentiment.polarity
    current_intensity = abs(current_polarity)
    
    #  STEP 8: Calculating Intensity Z-Score
    intensity_diff = current_intensity - bundle['REAL_INTENSITY_MEAN']
    intensity_z_score = intensity_diff / bundle['REAL_INTENSITY_STD']

    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    
    # AVERAGING BIAS AND POLARITY
    
    # STEP 9: Linguistic Score
    # Creating a combined deviation score (Bias + Intensity)
    # Bias and intensity are both taken into account by taking their average.
    avg_deviation = (max(0, bias_z_score) + max(0, intensity_z_score)) / 2


    # STEP 10: Scaling the deviation to a 0-100 penalty score.
    # a higher multiplier (60) for the internal risk calculation than the display score (40) to be more aggressive against misinformation.
    # 1.0 STD deviation = 60 point penalty (internal)
    # 1.66 STD deviation = 100 point penalty (internal)
    risk = min(100, avg_deviation * 60)
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------

    # STEP 11:
    
    # TONE EXPLANATION
    
    # Both the bias (subjectivity) and Intensity (Polarity) are considerd to categorize a text to a tone.
    # the bias and intensity Z-Scores are used in combination
    # bias z-score tells us how opinionated the text is and intensity z-score tells us how loud / emotional language is
    # compared TO AVERGAE POLARITY OF REAL-LABELED NEWS seen in the FineFake dataset
    
    # Using 0.4 as the threshold; any text more than 0.4 Standard Deviations away from the mean 
    # is flagged as "moving away from neutral" to ensure even subtle bias is caught.

    tone_threshold = 0.4
    abs_bias_z_score = abs(bias_z_score)
    abs_intensity_z_score = abs(intensity_z_score)

    if abs_bias_z_score > tone_threshold and abs_intensity_z_score > tone_threshold:
        tone_label = 'Highly Opinionated and Emotional'

    # 2. High in Bias OR High in Intensity (Catching the specific leans)

    elif abs_bias_z_score > tone_threshold:
        tone_label = 'Subtle Opinions and maybe Persuasive'

    elif abs_intensity_z_score > tone_threshold:
        tone_label = 'Dramatic / Sensationalist news'

    elif abs_bias_z_score <= tone_threshold and abs_intensity_z_score <= tone_threshold:
            # fewer opinions (less biased) + low emotional language (the IDEAL news standard)
        tone_label = 'Neutral / Standard Reporting'

        
    # -------------------------------------------------------------------------------------------------------------------------------------------------------

    # STEP 12:
    
    # RELIABILITY  
    
    # STEP 11:  Reliability 
    # Note: This is actually a content-based reliability score, but it can be used to derive conclusions about the source.
    # That interpretation is left upto the user. 
    # Simple formula: 100 minus weighted penalties. Penalty 'points' are deducted from 100 to get this score.
    # 2 penalties: 
                # 1. fact / bias check using fake likelihood score from ml model             (80% score from ML model)
                # 2. tone / style check using risk which combines bias and intensity         (20% from risk = Bias + Intensity combined)
    # tone check (risk) has lower weightage (20%) because a text can be a bit biased but still be factually true.
    # Bias and fake-likelihood score should not be interpreted together or based off each other.
    # True texts can have some bias, as it depends on the author and their writing-style.
    
    fact_penalty = fake_likelihood_score * 0.80
    risk_penalty = risk * 0.20
    
    final_reliability = 100 - fact_penalty - risk_penalty
    if final_reliability < 0:
        final_reliability = 0                   # To ensure score does NOT drop below 0 ever.
    if final_reliability > 100:
        final_reliability = 100                 # To ensure score does NOT cross 0 ever.
    
    # -------------------------------------------------------------------------------------------------------------------------------------------------------

    # STEP 13:
    
    # TOP-CONTRIBUTING WORDS

    # Get indices all words from array (created during vectorization) that appear in new text uploaded by user
    # Returns a tuple of (nonzero_row_index, nonzero_col_index)
    # Example: That is (array([0, 0, 1, 1, 2]), array([0, 2, 1, 2, 0]))
    # Non-zero is present at (0,0), (0,2)....etc.
    # here, the rows will be indices of texts from each row and columns will be the unique words / bigrams
    # SO rows not needed (first element of returned tuple not needed) 
    # Since you’re only uploading one text at a time for prediction, your vectorized_data has shape (1, n_features).
    # That means there’s only one row (row 0).
    # The first element of the tuple (row_indices) will just be [0, 0, 0, …].
    # The second element (col_indices) is the one that matters: it tells you which words from the vocabulary actually appear in this uploaded text.
    
    nonzero_indices = vectorized_data.nonzero()[1]
    feature_names = bundle['vectorizer'].get_feature_names_out()

    # Get SVM weights (coefficients) of linear
    # one weight assigned to each tfidf feature (each word / bigram chosen by vectorizer)
    # Positive weights = Real indicators, Negative weights = Fake indicators
    # negative represents one class, positive the other (for binary classification)
    weights = bundle['model'].coef_.toarray().flatten()
    
    scored_words = []
    for index in nonzero_indices:
        word = feature_names[index]
        weight = weights[index]
        scored_words.append((word, weight))
        
    # Sort: Lowest weights (Fake triggers) first, Highest weights (Real markers) last
    scored_words.sort(key= lambda x: x[1])
    
    top_fake_triggers = [w[0] for w in scored_words[:5]]    # lowest numbers (negative) imply fake label
    top_real_indicators = [w[0] for w in scored_words[-5:]]    # higher numbers (positive) imply real label
    # Selects first element of tuple (the word itself) for first 5 (most negative weights)
    # Selcts first element of tuple (word itself) for last 5 tuples (ones with most positive weights)

    # Use SVM weights, because it helps us see which words pushed the decision towards fake or real (in this case, the probability) the most

    return fake_likelihood_score, final_scaled_bias_score, final_reliability, tone_label, top_fake_triggers, top_real_indicators
