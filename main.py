import numpy as np
import pandas as pd
import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("AppGallery.csv")

# # Activity 1: Data selection
df['Interaction content'] = df['Interaction content'].values.astype('U')
df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')
df["y1"] = df["Type 1"]
df["y2"] = df["Type 2"]
df["y3"] = df["Type 3"]
df["y4"] = df["Type 4"]
df["x"] = df['Interaction content']
df["y"] = df["y2"]
df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
df.shape
print(df.shape)

temp = df

# Activity 3: Translation.py function
def trans_to_en(texts):
    t2t_m = "facebook/m2m100_418M"
    t2t_pipe = pipeline(task='text2text-generation', model=t2t_m)

    model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
    tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
    nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid",
                                 download_method=DownloadMethod.REUSE_RESOURCES)

    text_en_l = []
    for text in texts:
        if text == "":
            text_en_l.append(text)
            continue

        doc = nlp_stanza(text)
        if doc.lang == "en":
            text_en_l.append(text)
        else:
            lang = doc.lang
            tokenizer.src_lang = lang
            encoded_hi = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
            text_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            text_en_l.append(text_en[0])

    return text_en_l

# Activty 3 fucntion called
temp["ts_en"] = trans_to_en(temp["Ticket Summary"].to_list())

# Activity 4: Noise Removal
noise = "(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
temp["ts"] = temp["Ticket Summary"].str.lower().replace(noise, " ", regex=True).replace(r'\s+', ' ', regex=True).str.strip()

temp["ts"] = temp["ts"].str.lower().replace(noise, " ", regex=True).replace(r'\\s+', ' ',
                                                                                         regex=True).str.strip() 
temp_debug = temp.loc[:, ["Ticket Summary", "ts", "y"]]

temp["ic"] = temp["Interaction content"].str.lower()
noise_1 = [
        "(from :)|(subject :)|(sent :)|(r\\s*:)|(re\\s*:)",
        "(january|february|march|april|may|june|july|august|september|october|november|december)",
        "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        "\\d{2}(:|.)\\d{2}",
        "(xxxxx@xxxx\\.com)|(\\*{5}\\([a-z]+\\))",
        "dear ((customer)|(user))",
        "dear",
        "(hello)|(hallo)|(hi )|(hi there)",
        "good morning",
        "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
        "thank you for contacting us",
        "thank you for your availability",
        "thank you for providing us this information",
        "thank you for contacting",
        "thank you for reaching us (back)?",
        "thank you for patience",
        "thank you for (your)? reply",
        "thank you for (your)? response",
        "thank you for (your)? cooperation",
        "thank you for providing us with more information",
        "thank you very kindly",
        "thank you( very much)?",
        "i would like to follow up on the case you raised on the date",
        "i will do my very best to assist you"
        "in order to give you the best solution",
        "could you please clarify your request with following information:"
        "in this matter",
        "we hope you(( are)|('re)) doing ((fine)|(well))",
        "i would like to follow up on the case you raised on",
        "we apologize for the inconvenience",
        "sent from my huawei (cell )?phone",
        "original message",
        "customer support team",
        "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
        "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
        "canada, australia, new zealand and other countries",
        "\\d+",
        "[^0-9a-zA-Z]+",
        "(\\s|^).(\\s|$)"]
for noise in noise_1:
        # print(noise)
    temp["ic"] = temp["ic"].replace(noise, " ", regex=True)
    temp["ic"] = temp["ic"].replace(r'\\s+', ' ', regex=True).str.strip()
    temp_debug = temp.loc[:, ["Interaction content", "ic", "y"]]

    
    good_y1 = temp.y1.value_counts()[temp.y1.value_counts() > 10].index
    temp = temp.loc[temp.y1.isin(good_y1)]
    

# Activity 6: Text Representation in Numeric Form
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
x1 = tfidfconverter.fit_transform(temp["Interaction content"]).toarray()
x2 = tfidfconverter.fit_transform(temp["Ticket Summary"]).toarray()
X = np.concatenate((x1, x2), axis=1)

# Activty 7: Data Imbalance
y_series = pd.Series(y)
good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
y_good = y[y_series.isin(good_y_value)]
X_good = X[y_series.isin(good_y_value)]
y_bad = y[y_series.isin(good_y_value) == False]
X_bad = X[y_series.isin(good_y_value) == False]


# Activity 9: Data Preparation for Modeling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Activity 10: Model Selection
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)

# Activity 11: Training
classifier.fit(X_train, y_train)

# Testing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)

# Result Display
p_result = pd.DataFrame(classifier.predict_proba(X_test))
p_result.columns = classifier.classes_
print(p_result)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix Visual form
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()
