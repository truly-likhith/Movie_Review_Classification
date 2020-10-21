# Movie_Review_Classification

### NATURAL LANGUAGE PROCESSING
- Load input files and read reviews
- Tokenize
- Stopwords
- Stemming/lemmitization
- Write cleaned data to output file
- Bag of words
- TF-IDF(Term Frequency and Inverse document frequency)


```python
import spacy
from spacy import displacy
```


```python
!pip3 install -U spacy
!python3 -m spacy download en_core_web_sm
```

    Requirement already up-to-date: spacy in f:\anaconda\lib\site-packages (2.3.2)
    Requirement already satisfied, skipping upgrade: requests<3.0.0,>=2.13.0 in f:\anaconda\lib\site-packages (from spacy) (2.21.0)
    Requirement already satisfied, skipping upgrade: numpy>=1.15.0 in f:\anaconda\lib\site-packages (from spacy) (1.16.2)
    Requirement already satisfied, skipping upgrade: plac<1.2.0,>=0.9.6 in f:\anaconda\lib\site-packages (from spacy) (1.1.3)
    Requirement already satisfied, skipping upgrade: murmurhash<1.1.0,>=0.28.0 in f:\anaconda\lib\site-packages (from spacy) (1.0.2)
    Requirement already satisfied, skipping upgrade: preshed<3.1.0,>=3.0.2 in f:\anaconda\lib\site-packages (from spacy) (3.0.2)
    Requirement already satisfied, skipping upgrade: setuptools in f:\anaconda\lib\site-packages (from spacy) (40.8.0)
    Requirement already satisfied, skipping upgrade: cymem<2.1.0,>=2.0.2 in f:\anaconda\lib\site-packages (from spacy) (2.0.3)
    Requirement already satisfied, skipping upgrade: thinc==7.4.1 in f:\anaconda\lib\site-packages (from spacy) (7.4.1)
    Requirement already satisfied, skipping upgrade: wasabi<1.1.0,>=0.4.0 in f:\anaconda\lib\site-packages (from spacy) (0.8.0)
    Requirement already satisfied, skipping upgrade: catalogue<1.1.0,>=0.0.7 in f:\anaconda\lib\site-packages (from spacy) (1.0.0)
    Requirement already satisfied, skipping upgrade: tqdm<5.0.0,>=4.38.0 in f:\anaconda\lib\site-packages (from spacy) (4.50.2)
    Requirement already satisfied, skipping upgrade: srsly<1.1.0,>=1.0.2 in f:\anaconda\lib\site-packages (from spacy) (1.0.2)
    Requirement already satisfied, skipping upgrade: blis<0.5.0,>=0.4.0 in f:\anaconda\lib\site-packages (from spacy) (0.4.1)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in f:\anaconda\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2.8)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in f:\anaconda\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2020.6.20)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in f:\anaconda\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (3.0.4)
    Requirement already satisfied, skipping upgrade: urllib3<1.25,>=1.21.1 in f:\anaconda\lib\site-packages (from requests<3.0.0,>=2.13.0->spacy) (1.24.1)
    Requirement already satisfied, skipping upgrade: importlib-metadata>=0.20; python_version < "3.8" in f:\anaconda\lib\site-packages (from catalogue<1.1.0,>=0.0.7->spacy) (1.7.0)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in f:\anaconda\lib\site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy) (3.1.0)
    

    Python was not found but can be installed from the Microsoft Store: https://go.microsoft.com/fwlink?linkID=2082640
    


```python
import en_core_web_sm
```


```python
nlp = spacy.load('en_core_web_sm')
```


```python
import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
```

#### Lets get the basic understand before we to the project


```python
text = "Apple, This is first sentence. and Google this is another one. here 3rd one is"
```


```python
doc = nlp(text)
```


```python
doc
```




    Apple, This is first sentence. and Google this is another one. here 3rd one is



### Tokenize the each word


```python
for token in doc:
    print(token)
```

    Apple
    ,
    This
    is
    first
    sentence
    .
    and
    Google
    this
    is
    another
    one
    .
    here
    3rd
    one
    is
    

## Get the sentence in the above text 
- parser - which is the anlayising the syntatical analysis on symbols or either NLP ,COMPUTER LANGUAGE ETC..


```python
sent = nlp.create_pipe('sentencizer')
```


```python
nlp.add_pipe(sent, before='parser')
#Making the sentence syntatically format or structured
```


```python
doc = nlp(text)
```


```python
doc
```




    Apple, This is first sentence. and Google this is another one. here 3rd one is




```python
for sent in doc.sents:
    print(sent)
```

    Apple, This is first sentence.
    and Google this is another one.
    here 3rd one is
    

### REMOVE STOP_WORDS


```python
from spacy.lang.en.stop_words import STOP_WORDS
```


```python
stopwords = list(STOP_WORDS)
```


```python
stopwords
```




    ['hereupon',
     'who',
     'have',
     'regarding',
     'by',
     'either',
     '’m',
     'our',
     'since',
     'also',
     'say',
     'us',
     'name',
     'each',
     '‘ve',
     'somewhere',
     'itself',
     'whose',
     'namely',
     'your',
     'part',
     'whom',
     'many',
     'its',
     'nowhere',
     'with',
     'others',
     'it',
     'while',
     'whereafter',
     'mostly',
     'very',
     'one',
     'enough',
     'everything',
     'thereby',
     'had',
     'first',
     'within',
     'beforehand',
     "n't",
     'own',
     'hereby',
     'give',
     'if',
     'call',
     'am',
     'behind',
     'up',
     'over',
     'we',
     'you',
     'until',
     'nine',
     'out',
     'why',
     'but',
     'thru',
     'several',
     'not',
     "'m",
     'so',
     "'ll",
     'her',
     'fifteen',
     'everyone',
     'two',
     'seeming',
     'seems',
     'seemed',
     'although',
     'from',
     'done',
     'too',
     'should',
     'herein',
     'whither',
     'do',
     'just',
     'ourselves',
     'else',
     'before',
     'go',
     'twelve',
     'at',
     'when',
     'serious',
     'might',
     'elsewhere',
     'perhaps',
     'front',
     '’ll',
     '‘d',
     'neither',
     'doing',
     'still',
     'against',
     'here',
     "'re",
     'nor',
     'noone',
     'there',
     'third',
     'yours',
     'can',
     'these',
     'upon',
     'none',
     'under',
     'forty',
     'myself',
     'through',
     'how',
     'amount',
     'yet',
     'into',
     'then',
     'and',
     'because',
     'keep',
     'show',
     'anywhere',
     'already',
     'does',
     'what',
     'once',
     '‘s',
     'eleven',
     'twenty',
     'move',
     'everywhere',
     'above',
     'their',
     'fifty',
     'himself',
     'therein',
     'an',
     'never',
     'formerly',
     'further',
     'last',
     'get',
     'least',
     'much',
     'often',
     'are',
     'thence',
     'those',
     'together',
     'used',
     'thereafter',
     'few',
     'make',
     'whereupon',
     'anyway',
     'being',
     'anyone',
     'must',
     'during',
     "'d",
     'where',
     'something',
     'bottom',
     'every',
     'a',
     'well',
     'someone',
     'whole',
     'for',
     'which',
     'herself',
     'made',
     'me',
     'wherein',
     'some',
     'on',
     'next',
     'most',
     'nevertheless',
     'whence',
     'onto',
     'five',
     'moreover',
     'becomes',
     'him',
     'nobody',
     'did',
     'he',
     'could',
     'various',
     'toward',
     'was',
     'full',
     'be',
     're',
     'throughout',
     'almost',
     'back',
     'amongst',
     'meanwhile',
     'towards',
     'n‘t',
     'same',
     'ours',
     'top',
     'thereupon',
     'side',
     'become',
     'more',
     'around',
     'take',
     'sixty',
     'whoever',
     'among',
     'other',
     'really',
     'yourself',
     'whatever',
     'only',
     'in',
     'empty',
     'without',
     'nothing',
     'anything',
     'ten',
     'across',
     'is',
     'six',
     'along',
     'all',
     'three',
     'beside',
     '’ve',
     'after',
     'hundred',
     'no',
     'whether',
     'hence',
     'afterwards',
     'has',
     'sometimes',
     "'s",
     'latter',
     'she',
     'alone',
     'eight',
     'however',
     'whereby',
     'wherever',
     'even',
     'otherwise',
     'yourselves',
     'thus',
     'this',
     'now',
     'therefore',
     '‘re',
     'unless',
     'seem',
     'hers',
     '’re',
     'my',
     'sometime',
     'see',
     'via',
     'latterly',
     'of',
     'indeed',
     'whereas',
     'ca',
     'below',
     'i',
     'rather',
     'that',
     'whenever',
     'using',
     'beyond',
     'again',
     'than',
     'them',
     'the',
     '‘m',
     "'ve",
     'to',
     'please',
     'any',
     'or',
     'between',
     'his',
     'less',
     'n’t',
     'hereafter',
     'will',
     'anyhow',
     '’s',
     'they',
     'as',
     'off',
     'due',
     'about',
     'somehow',
     'themselves',
     'besides',
     'always',
     'another',
     'four',
     'former',
     '’d',
     'been',
     'both',
     'cannot',
     'quite',
     'per',
     'becoming',
     'down',
     'became',
     'though',
     'were',
     'mine',
     '‘ll',
     'except',
     'may',
     'ever',
     'would',
     'such',
     'put']




```python
len(stopwords)
```




    326




```python
# removing stop words in our sentence
for token in doc:
    if token.is_stop == False:
        print(token)
```

    Apple
    ,
    sentence
    .
    Google
    .
    3rd
    

## LEMMATIZATION


```python
doc = nlp('run runs running runner')

```


```python
for lem in doc:
    print(lem.text,lem.lemma_)
```

    run run
    runs run
    running run
    runner runner
    

### POS(PARTS OF SPEECH)


```python
doc = nlp('All is well at your end!')
```


```python
for token in doc:
    print(token.text,token.pos_)
```

    All DET
    is AUX
    well ADJ
    at ADP
    your DET
    end NOUN
    ! PUNCT
    


```python
displacy.render(doc, style = 'dep')
```


<span class="tex2jax_ignore"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="en" id="023300c2657745d0b470d9f14d58da75-0" class="displacy" width="1100" height="312.0" direction="ltr" style="max-width: none; height: 312.0px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr">
<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="50">All</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">DET</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="225">is</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="225">AUX</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="400">well</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="400">ADJ</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="575">at</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="575">ADP</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="750">your</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="750">DET</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="925">end!</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="925">NOUN</tspan>
</text>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-023300c2657745d0b470d9f14d58da75-0-0" stroke-width="2px" d="M70,177.0 C70,89.5 220.0,89.5 220.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-023300c2657745d0b470d9f14d58da75-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nsubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M70,179.0 L62,167.0 78,167.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-023300c2657745d0b470d9f14d58da75-0-1" stroke-width="2px" d="M245,177.0 C245,89.5 395.0,89.5 395.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-023300c2657745d0b470d9f14d58da75-0-1" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">advmod</textPath>
    </text>
    <path class="displacy-arrowhead" d="M395.0,179.0 L403.0,167.0 387.0,167.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-023300c2657745d0b470d9f14d58da75-0-2" stroke-width="2px" d="M245,177.0 C245,2.0 575.0,2.0 575.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-023300c2657745d0b470d9f14d58da75-0-2" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">prep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M575.0,179.0 L583.0,167.0 567.0,167.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-023300c2657745d0b470d9f14d58da75-0-3" stroke-width="2px" d="M770,177.0 C770,89.5 920.0,89.5 920.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-023300c2657745d0b470d9f14d58da75-0-3" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">poss</textPath>
    </text>
    <path class="displacy-arrowhead" d="M770,179.0 L762,167.0 778,167.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-023300c2657745d0b470d9f14d58da75-0-4" stroke-width="2px" d="M595,177.0 C595,2.0 925.0,2.0 925.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-023300c2657745d0b470d9f14d58da75-0-4" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">pobj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M925.0,179.0 L933.0,167.0 917.0,167.0" fill="currentColor"/>
</g>
</svg></span>


### ENTITY DETECTION


```python
doc = nlp("New York City on Tuesday declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases. At least 285 people have contracted measles in the city since September, mostly in Brooklyn’s Williamsburg neighborhood. The order covers four Zip codes there, Mayor Bill de Blasio (D) said Tuesday. The mandate orders all unvaccinated people in the area, including a concentration of Orthodox Jews, to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined up to $1,000.")
```


```python
doc
```




    New York City on Tuesday declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases. At least 285 people have contracted measles in the city since September, mostly in Brooklyn’s Williamsburg neighborhood. The order covers four Zip codes there, Mayor Bill de Blasio (D) said Tuesday. The mandate orders all unvaccinated people in the area, including a concentration of Orthodox Jews, to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined up to $1,000.




```python
displacy.render(doc, style = 'ent')
```


<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    New York City
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 on 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Tuesday
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases. 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    At least 285
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 people have contracted measles in the city since 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    September
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
, mostly in 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Brooklyn
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
’s 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Williamsburg
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 neighborhood. The order covers 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    four
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 Zip codes there, Mayor 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Bill de Blasio
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 (D) said 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Tuesday
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
. The mandate orders all unvaccinated people in the area, including a concentration of 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Orthodox
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
 
<mark class="entity" style="background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Jews
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">NORP</span>
</mark>
, to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    up to $1,000
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">MONEY</span>
</mark>
.</div></span>


####                           ** PROJECT ** 
####       ** TEXT CLASSIFICATION (MOVIE REVIEW IMDB AND AMAZON DATASET) **


```python
import pandas as pd
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```


```python
data_yelp = pd.read_csv('NLP(folders)/yelp_label.txt',sep='\t',header=None)
```


```python
data_yelp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wow... Loved this place.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crust is not good.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not tasty and the texture was just nasty.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stopped by during the late May bank holiday of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The selection on the menu was great and so wer...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
column_name = ['Review','Sentiment']
data_yelp.columns = column_name
```


```python
data_yelp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wow... Loved this place.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crust is not good.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not tasty and the texture was just nasty.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stopped by during the late May bank holiday of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The selection on the menu was great and so wer...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_yelp.shape
```




    (1000, 2)




```python
data_yelp.columns
```




    Index(['Review', 'Sentiment'], dtype='object')



##### LOAD AMAZON DATASET


```python
data_amazon = pd.read_csv('NLP(folders)/amazon_labeled.txt',sep='\t',header=None)
```


```python
data_amazon.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>So there is no way for me to plug it in here i...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Good case, Excellent value.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Great for the jawbone.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tied to charger for conversations lasting more...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The mic is great.</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_amazon.columns = column_name
```


```python
data_amazon.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>So there is no way for me to plug it in here i...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Good case, Excellent value.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Great for the jawbone.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tied to charger for conversations lasting more...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The mic is great.</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_amazon.shape
```




    (1000, 2)



### IMDB DATA_LOAD


```python
data_imdb = pd.read_csv('NLP(folders)/imdb.txt', sep = '\t', header = None)
```


```python
data_imdb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A very, very, very slow-moving, aimless movie ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Not sure who was more lost - the flat characte...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Attempting artiness with black &amp; white and cle...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Very little music or anything to speak of.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The best scene in the movie was when Gerardo i...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_imdb.columns = column_name
```


```python
data_imdb.shape
```




    (748, 2)




```python
data_imdb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A very, very, very slow-moving, aimless movie ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Not sure who was more lost - the flat characte...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Attempting artiness with black &amp; white and cle...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Very little music or anything to speak of.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The best scene in the movie was when Gerardo i...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_yelp.shape
```




    (1000, 2)




```python
data_imdb.shape
```




    (748, 2)




```python
data_amazon.shape
```




    (1000, 2)



### We have 'three' type of data now
-  Now append all the data in a single format


```python
data = data_yelp.append([data_imdb,data_amazon],ignore_index=True)
```


```python
data.shape
```




    (2748, 2)




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wow... Loved this place.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crust is not good.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Not tasty and the texture was just nasty.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stopped by during the late May bank holiday of...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The selection on the menu was great and so wer...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Lets check how many positive and negative values in the data
- 1 --> 1386 (+ve reviews)
- 0 --> 1362 (-ve reviews)


```python
data['Sentiment'].value_counts()
```




    1    1386
    0    1362
    Name: Sentiment, dtype: int64




```python
## check the null values
data.isnull().sum()

# o/p : no null values in the data
```




    Review       0
    Sentiment    0
    dtype: int64



### TOKENIZATION


```python
import string
```


```python
pun = string.punctuation
```


```python
pun
```




    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'



- LOWE THE PUNCTUATION 
- REMOVE STOP WORDS AND NON-PUNCTUATION WORDS TO THE ARRAY
- FINALLY PRINT THE RESULT


```python
def text_cleaning_data(sentence):
    doc = nlp(sentence)
    # LEMMATIZATION
    tokens = []
    for token in doc:
        if token.lemma_!= "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    # STOP WORDS
    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in pun:
            cleaned_tokens.append(token)
    return cleaned_tokens
```


```python
text_cleaning_data("         Hello How are you guys, please like my presentation")
```




    ['hello', 'guy', 'like', 'presentation']



### VECTORIZATION ENGINEERING(TF-IDF)


```python
from sklearn.svm import LinearSVC
```


```python
tfidf = TfidfVectorizer(tokenizer = text_cleaning_data)
classifier = LinearSVC()
#it willm be done according to function(text_cleaning_data)
```


```python
X = data['Review']
y = data['Sentiment']

```


```python
X.shape
```




    (2748,)




```python
y.shape
```




    (2748,)




```python
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)
```


```python
X_train.shape
```




    (2198,)




```python
X_test.shape
```




    (550,)




```python
y_train.shape
```




    (2198,)




```python
y_test.shape
```




    (550,)




```python
clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])
```


```python
clf.fit(X_train,y_train)
```




    Pipeline(memory=None,
         steps=[('tfidf', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.float64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,...ax_iter=1000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0))])




```python
print("Our model is scoring: ",clf.score(X_train,y_train)*100)
```

    Our model is scoring:  97.49772520473158
    


```python
y_pred = clf.predict(X_test)
```


```python
y_pred
```




    array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1,
           0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1,
           1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1,
           1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
           0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1,
           0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1,
           0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0,
           1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
           1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1,
           1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
           1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1,
           1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
           1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
           1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
           1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1,
           0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0,
           0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0,
           1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
           1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
           0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
          dtype=int64)




```python
confusion_matrix(y_test, y_pred)
```




    array([[212,  67],
           [ 61, 210]], dtype=int64)




```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.78      0.76      0.77       279
               1       0.76      0.77      0.77       271
    
       micro avg       0.77      0.77      0.77       550
       macro avg       0.77      0.77      0.77       550
    weighted avg       0.77      0.77      0.77       550
    
    


```python
clf.predict(['Wow, this is bad'])
```




    array([0], dtype=int64)




```python
clf.predict(['Worth of watching it. Please like it'])
```




    array([1], dtype=int64)




```python
clf.predict(['Wow, this is amzing lesson i liked it'])
```




    array([1], dtype=int64)




```python
clf.predict(['i hate this movie'])
```




    array([0], dtype=int64)




```python
clf.predict(['i hate this kind of movies but this is an exemption but i like this a little bit'])
```




    array([0], dtype=int64)




```python
clf.predict(['fantastic movie'])
```




    array([1], dtype=int64)




```python

```
