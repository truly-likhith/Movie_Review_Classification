# Movie_Review_Classification
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### NATURAL LANGUAGE PROCESSING\n",
    "- Load input files and read reviews\n",
    "- Tokenize\n",
    "- Stopwords\n",
    "- Stemming/lemmitization\n",
    "- Write cleaned data to output file\n",
    "- Bag of words\n",
    "- TF-IDF(Term Frequency and Inverse document frequency)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy\n",
    "from spacy import displacy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already up-to-date: spacy in f:\\anaconda\\lib\\site-packages (2.3.2)\n",
      "Requirement already satisfied, skipping upgrade: requests<3.0.0,>=2.13.0 in f:\\anaconda\\lib\\site-packages (from spacy) (2.21.0)\n",
      "Requirement already satisfied, skipping upgrade: numpy>=1.15.0 in f:\\anaconda\\lib\\site-packages (from spacy) (1.16.2)\n",
      "Requirement already satisfied, skipping upgrade: plac<1.2.0,>=0.9.6 in f:\\anaconda\\lib\\site-packages (from spacy) (1.1.3)\n",
      "Requirement already satisfied, skipping upgrade: murmurhash<1.1.0,>=0.28.0 in f:\\anaconda\\lib\\site-packages (from spacy) (1.0.2)\n",
      "Requirement already satisfied, skipping upgrade: preshed<3.1.0,>=3.0.2 in f:\\anaconda\\lib\\site-packages (from spacy) (3.0.2)\n",
      "Requirement already satisfied, skipping upgrade: setuptools in f:\\anaconda\\lib\\site-packages (from spacy) (40.8.0)\n",
      "Requirement already satisfied, skipping upgrade: cymem<2.1.0,>=2.0.2 in f:\\anaconda\\lib\\site-packages (from spacy) (2.0.3)\n",
      "Requirement already satisfied, skipping upgrade: thinc==7.4.1 in f:\\anaconda\\lib\\site-packages (from spacy) (7.4.1)\n",
      "Requirement already satisfied, skipping upgrade: wasabi<1.1.0,>=0.4.0 in f:\\anaconda\\lib\\site-packages (from spacy) (0.8.0)\n",
      "Requirement already satisfied, skipping upgrade: catalogue<1.1.0,>=0.0.7 in f:\\anaconda\\lib\\site-packages (from spacy) (1.0.0)\n",
      "Requirement already satisfied, skipping upgrade: tqdm<5.0.0,>=4.38.0 in f:\\anaconda\\lib\\site-packages (from spacy) (4.50.2)\n",
      "Requirement already satisfied, skipping upgrade: srsly<1.1.0,>=1.0.2 in f:\\anaconda\\lib\\site-packages (from spacy) (1.0.2)\n",
      "Requirement already satisfied, skipping upgrade: blis<0.5.0,>=0.4.0 in f:\\anaconda\\lib\\site-packages (from spacy) (0.4.1)\n",
      "Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in f:\\anaconda\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2.8)\n",
      "Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in f:\\anaconda\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2020.6.20)\n",
      "Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in f:\\anaconda\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (3.0.4)\n",
      "Requirement already satisfied, skipping upgrade: urllib3<1.25,>=1.21.1 in f:\\anaconda\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (1.24.1)\n",
      "Requirement already satisfied, skipping upgrade: importlib-metadata>=0.20; python_version < \"3.8\" in f:\\anaconda\\lib\\site-packages (from catalogue<1.1.0,>=0.0.7->spacy) (1.7.0)\n",
      "Requirement already satisfied, skipping upgrade: zipp>=0.5 in f:\\anaconda\\lib\\site-packages (from importlib-metadata>=0.20; python_version < \"3.8\"->catalogue<1.1.0,>=0.0.7->spacy) (3.1.0)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Python was not found but can be installed from the Microsoft Store: https://go.microsoft.com/fwlink?linkID=2082640\n"
     ]
    }
   ],
   "source": [
    "!pip3 install -U spacy\n",
    "!python3 -m spacy download en_core_web_sm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import en_core_web_sm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "nlp = spacy.load('en_core_web_sm')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy\n",
    "spacy.prefer_gpu()\n",
    "nlp = spacy.load(\"en_core_web_sm\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Lets get the basic understand before we to the project"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "text = \"Apple, This is first sentence. and Google this is another one. here 3rd one is\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "doc = nlp(text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Apple, This is first sentence. and Google this is another one. here 3rd one is"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "doc"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Tokenize the each word"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Apple\n",
      ",\n",
      "This\n",
      "is\n",
      "first\n",
      "sentence\n",
      ".\n",
      "and\n",
      "Google\n",
      "this\n",
      "is\n",
      "another\n",
      "one\n",
      ".\n",
      "here\n",
      "3rd\n",
      "one\n",
      "is\n"
     ]
    }
   ],
   "source": [
    "for token in doc:\n",
    "    print(token)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Get the sentence in the above text \n",
    "- parser - which is the anlayising the syntatical analysis on symbols or either NLP ,COMPUTER LANGUAGE ETC.."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "sent = nlp.create_pipe('sentencizer')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "nlp.add_pipe(sent, before='parser')\n",
    "#Making the sentence syntatically format or structured"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "doc = nlp(text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Apple, This is first sentence. and Google this is another one. here 3rd one is"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "doc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Apple, This is first sentence.\n",
      "and Google this is another one.\n",
      "here 3rd one is\n"
     ]
    }
   ],
   "source": [
    "for sent in doc.sents:\n",
    "    print(sent)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### REMOVE STOP_WORDS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "from spacy.lang.en.stop_words import STOP_WORDS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "stopwords = list(STOP_WORDS)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['hereupon',\n",
       " 'who',\n",
       " 'have',\n",
       " 'regarding',\n",
       " 'by',\n",
       " 'either',\n",
       " '’m',\n",
       " 'our',\n",
       " 'since',\n",
       " 'also',\n",
       " 'say',\n",
       " 'us',\n",
       " 'name',\n",
       " 'each',\n",
       " '‘ve',\n",
       " 'somewhere',\n",
       " 'itself',\n",
       " 'whose',\n",
       " 'namely',\n",
       " 'your',\n",
       " 'part',\n",
       " 'whom',\n",
       " 'many',\n",
       " 'its',\n",
       " 'nowhere',\n",
       " 'with',\n",
       " 'others',\n",
       " 'it',\n",
       " 'while',\n",
       " 'whereafter',\n",
       " 'mostly',\n",
       " 'very',\n",
       " 'one',\n",
       " 'enough',\n",
       " 'everything',\n",
       " 'thereby',\n",
       " 'had',\n",
       " 'first',\n",
       " 'within',\n",
       " 'beforehand',\n",
       " \"n't\",\n",
       " 'own',\n",
       " 'hereby',\n",
       " 'give',\n",
       " 'if',\n",
       " 'call',\n",
       " 'am',\n",
       " 'behind',\n",
       " 'up',\n",
       " 'over',\n",
       " 'we',\n",
       " 'you',\n",
       " 'until',\n",
       " 'nine',\n",
       " 'out',\n",
       " 'why',\n",
       " 'but',\n",
       " 'thru',\n",
       " 'several',\n",
       " 'not',\n",
       " \"'m\",\n",
       " 'so',\n",
       " \"'ll\",\n",
       " 'her',\n",
       " 'fifteen',\n",
       " 'everyone',\n",
       " 'two',\n",
       " 'seeming',\n",
       " 'seems',\n",
       " 'seemed',\n",
       " 'although',\n",
       " 'from',\n",
       " 'done',\n",
       " 'too',\n",
       " 'should',\n",
       " 'herein',\n",
       " 'whither',\n",
       " 'do',\n",
       " 'just',\n",
       " 'ourselves',\n",
       " 'else',\n",
       " 'before',\n",
       " 'go',\n",
       " 'twelve',\n",
       " 'at',\n",
       " 'when',\n",
       " 'serious',\n",
       " 'might',\n",
       " 'elsewhere',\n",
       " 'perhaps',\n",
       " 'front',\n",
       " '’ll',\n",
       " '‘d',\n",
       " 'neither',\n",
       " 'doing',\n",
       " 'still',\n",
       " 'against',\n",
       " 'here',\n",
       " \"'re\",\n",
       " 'nor',\n",
       " 'noone',\n",
       " 'there',\n",
       " 'third',\n",
       " 'yours',\n",
       " 'can',\n",
       " 'these',\n",
       " 'upon',\n",
       " 'none',\n",
       " 'under',\n",
       " 'forty',\n",
       " 'myself',\n",
       " 'through',\n",
       " 'how',\n",
       " 'amount',\n",
       " 'yet',\n",
       " 'into',\n",
       " 'then',\n",
       " 'and',\n",
       " 'because',\n",
       " 'keep',\n",
       " 'show',\n",
       " 'anywhere',\n",
       " 'already',\n",
       " 'does',\n",
       " 'what',\n",
       " 'once',\n",
       " '‘s',\n",
       " 'eleven',\n",
       " 'twenty',\n",
       " 'move',\n",
       " 'everywhere',\n",
       " 'above',\n",
       " 'their',\n",
       " 'fifty',\n",
       " 'himself',\n",
       " 'therein',\n",
       " 'an',\n",
       " 'never',\n",
       " 'formerly',\n",
       " 'further',\n",
       " 'last',\n",
       " 'get',\n",
       " 'least',\n",
       " 'much',\n",
       " 'often',\n",
       " 'are',\n",
       " 'thence',\n",
       " 'those',\n",
       " 'together',\n",
       " 'used',\n",
       " 'thereafter',\n",
       " 'few',\n",
       " 'make',\n",
       " 'whereupon',\n",
       " 'anyway',\n",
       " 'being',\n",
       " 'anyone',\n",
       " 'must',\n",
       " 'during',\n",
       " \"'d\",\n",
       " 'where',\n",
       " 'something',\n",
       " 'bottom',\n",
       " 'every',\n",
       " 'a',\n",
       " 'well',\n",
       " 'someone',\n",
       " 'whole',\n",
       " 'for',\n",
       " 'which',\n",
       " 'herself',\n",
       " 'made',\n",
       " 'me',\n",
       " 'wherein',\n",
       " 'some',\n",
       " 'on',\n",
       " 'next',\n",
       " 'most',\n",
       " 'nevertheless',\n",
       " 'whence',\n",
       " 'onto',\n",
       " 'five',\n",
       " 'moreover',\n",
       " 'becomes',\n",
       " 'him',\n",
       " 'nobody',\n",
       " 'did',\n",
       " 'he',\n",
       " 'could',\n",
       " 'various',\n",
       " 'toward',\n",
       " 'was',\n",
       " 'full',\n",
       " 'be',\n",
       " 're',\n",
       " 'throughout',\n",
       " 'almost',\n",
       " 'back',\n",
       " 'amongst',\n",
       " 'meanwhile',\n",
       " 'towards',\n",
       " 'n‘t',\n",
       " 'same',\n",
       " 'ours',\n",
       " 'top',\n",
       " 'thereupon',\n",
       " 'side',\n",
       " 'become',\n",
       " 'more',\n",
       " 'around',\n",
       " 'take',\n",
       " 'sixty',\n",
       " 'whoever',\n",
       " 'among',\n",
       " 'other',\n",
       " 'really',\n",
       " 'yourself',\n",
       " 'whatever',\n",
       " 'only',\n",
       " 'in',\n",
       " 'empty',\n",
       " 'without',\n",
       " 'nothing',\n",
       " 'anything',\n",
       " 'ten',\n",
       " 'across',\n",
       " 'is',\n",
       " 'six',\n",
       " 'along',\n",
       " 'all',\n",
       " 'three',\n",
       " 'beside',\n",
       " '’ve',\n",
       " 'after',\n",
       " 'hundred',\n",
       " 'no',\n",
       " 'whether',\n",
       " 'hence',\n",
       " 'afterwards',\n",
       " 'has',\n",
       " 'sometimes',\n",
       " \"'s\",\n",
       " 'latter',\n",
       " 'she',\n",
       " 'alone',\n",
       " 'eight',\n",
       " 'however',\n",
       " 'whereby',\n",
       " 'wherever',\n",
       " 'even',\n",
       " 'otherwise',\n",
       " 'yourselves',\n",
       " 'thus',\n",
       " 'this',\n",
       " 'now',\n",
       " 'therefore',\n",
       " '‘re',\n",
       " 'unless',\n",
       " 'seem',\n",
       " 'hers',\n",
       " '’re',\n",
       " 'my',\n",
       " 'sometime',\n",
       " 'see',\n",
       " 'via',\n",
       " 'latterly',\n",
       " 'of',\n",
       " 'indeed',\n",
       " 'whereas',\n",
       " 'ca',\n",
       " 'below',\n",
       " 'i',\n",
       " 'rather',\n",
       " 'that',\n",
       " 'whenever',\n",
       " 'using',\n",
       " 'beyond',\n",
       " 'again',\n",
       " 'than',\n",
       " 'them',\n",
       " 'the',\n",
       " '‘m',\n",
       " \"'ve\",\n",
       " 'to',\n",
       " 'please',\n",
       " 'any',\n",
       " 'or',\n",
       " 'between',\n",
       " 'his',\n",
       " 'less',\n",
       " 'n’t',\n",
       " 'hereafter',\n",
       " 'will',\n",
       " 'anyhow',\n",
       " '’s',\n",
       " 'they',\n",
       " 'as',\n",
       " 'off',\n",
       " 'due',\n",
       " 'about',\n",
       " 'somehow',\n",
       " 'themselves',\n",
       " 'besides',\n",
       " 'always',\n",
       " 'another',\n",
       " 'four',\n",
       " 'former',\n",
       " '’d',\n",
       " 'been',\n",
       " 'both',\n",
       " 'cannot',\n",
       " 'quite',\n",
       " 'per',\n",
       " 'becoming',\n",
       " 'down',\n",
       " 'became',\n",
       " 'though',\n",
       " 'were',\n",
       " 'mine',\n",
       " '‘ll',\n",
       " 'except',\n",
       " 'may',\n",
       " 'ever',\n",
       " 'would',\n",
       " 'such',\n",
       " 'put']"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "stopwords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "326"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(stopwords)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Apple\n",
      ",\n",
      "sentence\n",
      ".\n",
      "Google\n",
      ".\n",
      "3rd\n"
     ]
    }
   ],
   "source": [
    "# removing stop words in our sentence\n",
    "for token in doc:\n",
    "    if token.is_stop == False:\n",
    "        print(token)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## LEMMATIZATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "doc = nlp('run runs running runner')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "run run\n",
      "runs run\n",
      "running run\n",
      "runner runner\n"
     ]
    }
   ],
   "source": [
    "for lem in doc:\n",
    "    print(lem.text,lem.lemma_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### POS(PARTS OF SPEECH)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "doc = nlp('All is well at your end!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "All DET\n",
      "is AUX\n",
      "well ADJ\n",
      "at ADP\n",
      "your DET\n",
      "end NOUN\n",
      "! PUNCT\n"
     ]
    }
   ],
   "source": [
    "for token in doc:\n",
    "    print(token.text,token.pos_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<span class=\"tex2jax_ignore\"><svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" xml:lang=\"en\" id=\"023300c2657745d0b470d9f14d58da75-0\" class=\"displacy\" width=\"1100\" height=\"312.0\" direction=\"ltr\" style=\"max-width: none; height: 312.0px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr\">\n",
       "<text class=\"displacy-token\" fill=\"currentColor\" text-anchor=\"middle\" y=\"222.0\">\n",
       "    <tspan class=\"displacy-word\" fill=\"currentColor\" x=\"50\">All</tspan>\n",
       "    <tspan class=\"displacy-tag\" dy=\"2em\" fill=\"currentColor\" x=\"50\">DET</tspan>\n",
       "</text>\n",
       "\n",
       "<text class=\"displacy-token\" fill=\"currentColor\" text-anchor=\"middle\" y=\"222.0\">\n",
       "    <tspan class=\"displacy-word\" fill=\"currentColor\" x=\"225\">is</tspan>\n",
       "    <tspan class=\"displacy-tag\" dy=\"2em\" fill=\"currentColor\" x=\"225\">AUX</tspan>\n",
       "</text>\n",
       "\n",
       "<text class=\"displacy-token\" fill=\"currentColor\" text-anchor=\"middle\" y=\"222.0\">\n",
       "    <tspan class=\"displacy-word\" fill=\"currentColor\" x=\"400\">well</tspan>\n",
       "    <tspan class=\"displacy-tag\" dy=\"2em\" fill=\"currentColor\" x=\"400\">ADJ</tspan>\n",
       "</text>\n",
       "\n",
       "<text class=\"displacy-token\" fill=\"currentColor\" text-anchor=\"middle\" y=\"222.0\">\n",
       "    <tspan class=\"displacy-word\" fill=\"currentColor\" x=\"575\">at</tspan>\n",
       "    <tspan class=\"displacy-tag\" dy=\"2em\" fill=\"currentColor\" x=\"575\">ADP</tspan>\n",
       "</text>\n",
       "\n",
       "<text class=\"displacy-token\" fill=\"currentColor\" text-anchor=\"middle\" y=\"222.0\">\n",
       "    <tspan class=\"displacy-word\" fill=\"currentColor\" x=\"750\">your</tspan>\n",
       "    <tspan class=\"displacy-tag\" dy=\"2em\" fill=\"currentColor\" x=\"750\">DET</tspan>\n",
       "</text>\n",
       "\n",
       "<text class=\"displacy-token\" fill=\"currentColor\" text-anchor=\"middle\" y=\"222.0\">\n",
       "    <tspan class=\"displacy-word\" fill=\"currentColor\" x=\"925\">end!</tspan>\n",
       "    <tspan class=\"displacy-tag\" dy=\"2em\" fill=\"currentColor\" x=\"925\">NOUN</tspan>\n",
       "</text>\n",
       "\n",
       "<g class=\"displacy-arrow\">\n",
       "    <path class=\"displacy-arc\" id=\"arrow-023300c2657745d0b470d9f14d58da75-0-0\" stroke-width=\"2px\" d=\"M70,177.0 C70,89.5 220.0,89.5 220.0,177.0\" fill=\"none\" stroke=\"currentColor\"/>\n",
       "    <text dy=\"1.25em\" style=\"font-size: 0.8em; letter-spacing: 1px\">\n",
       "        <textPath xlink:href=\"#arrow-023300c2657745d0b470d9f14d58da75-0-0\" class=\"displacy-label\" startOffset=\"50%\" side=\"left\" fill=\"currentColor\" text-anchor=\"middle\">nsubj</textPath>\n",
       "    </text>\n",
       "    <path class=\"displacy-arrowhead\" d=\"M70,179.0 L62,167.0 78,167.0\" fill=\"currentColor\"/>\n",
       "</g>\n",
       "\n",
       "<g class=\"displacy-arrow\">\n",
       "    <path class=\"displacy-arc\" id=\"arrow-023300c2657745d0b470d9f14d58da75-0-1\" stroke-width=\"2px\" d=\"M245,177.0 C245,89.5 395.0,89.5 395.0,177.0\" fill=\"none\" stroke=\"currentColor\"/>\n",
       "    <text dy=\"1.25em\" style=\"font-size: 0.8em; letter-spacing: 1px\">\n",
       "        <textPath xlink:href=\"#arrow-023300c2657745d0b470d9f14d58da75-0-1\" class=\"displacy-label\" startOffset=\"50%\" side=\"left\" fill=\"currentColor\" text-anchor=\"middle\">advmod</textPath>\n",
       "    </text>\n",
       "    <path class=\"displacy-arrowhead\" d=\"M395.0,179.0 L403.0,167.0 387.0,167.0\" fill=\"currentColor\"/>\n",
       "</g>\n",
       "\n",
       "<g class=\"displacy-arrow\">\n",
       "    <path class=\"displacy-arc\" id=\"arrow-023300c2657745d0b470d9f14d58da75-0-2\" stroke-width=\"2px\" d=\"M245,177.0 C245,2.0 575.0,2.0 575.0,177.0\" fill=\"none\" stroke=\"currentColor\"/>\n",
       "    <text dy=\"1.25em\" style=\"font-size: 0.8em; letter-spacing: 1px\">\n",
       "        <textPath xlink:href=\"#arrow-023300c2657745d0b470d9f14d58da75-0-2\" class=\"displacy-label\" startOffset=\"50%\" side=\"left\" fill=\"currentColor\" text-anchor=\"middle\">prep</textPath>\n",
       "    </text>\n",
       "    <path class=\"displacy-arrowhead\" d=\"M575.0,179.0 L583.0,167.0 567.0,167.0\" fill=\"currentColor\"/>\n",
       "</g>\n",
       "\n",
       "<g class=\"displacy-arrow\">\n",
       "    <path class=\"displacy-arc\" id=\"arrow-023300c2657745d0b470d9f14d58da75-0-3\" stroke-width=\"2px\" d=\"M770,177.0 C770,89.5 920.0,89.5 920.0,177.0\" fill=\"none\" stroke=\"currentColor\"/>\n",
       "    <text dy=\"1.25em\" style=\"font-size: 0.8em; letter-spacing: 1px\">\n",
       "        <textPath xlink:href=\"#arrow-023300c2657745d0b470d9f14d58da75-0-3\" class=\"displacy-label\" startOffset=\"50%\" side=\"left\" fill=\"currentColor\" text-anchor=\"middle\">poss</textPath>\n",
       "    </text>\n",
       "    <path class=\"displacy-arrowhead\" d=\"M770,179.0 L762,167.0 778,167.0\" fill=\"currentColor\"/>\n",
       "</g>\n",
       "\n",
       "<g class=\"displacy-arrow\">\n",
       "    <path class=\"displacy-arc\" id=\"arrow-023300c2657745d0b470d9f14d58da75-0-4\" stroke-width=\"2px\" d=\"M595,177.0 C595,2.0 925.0,2.0 925.0,177.0\" fill=\"none\" stroke=\"currentColor\"/>\n",
       "    <text dy=\"1.25em\" style=\"font-size: 0.8em; letter-spacing: 1px\">\n",
       "        <textPath xlink:href=\"#arrow-023300c2657745d0b470d9f14d58da75-0-4\" class=\"displacy-label\" startOffset=\"50%\" side=\"left\" fill=\"currentColor\" text-anchor=\"middle\">pobj</textPath>\n",
       "    </text>\n",
       "    <path class=\"displacy-arrowhead\" d=\"M925.0,179.0 L933.0,167.0 917.0,167.0\" fill=\"currentColor\"/>\n",
       "</g>\n",
       "</svg></span>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "displacy.render(doc, style = 'dep')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### ENTITY DETECTION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "doc = nlp(\"New York City on Tuesday declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases. At least 285 people have contracted measles in the city since September, mostly in Brooklyn’s Williamsburg neighborhood. The order covers four Zip codes there, Mayor Bill de Blasio (D) said Tuesday. The mandate orders all unvaccinated people in the area, including a concentration of Orthodox Jews, to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined up to $1,000.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "New York City on Tuesday declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases. At least 285 people have contracted measles in the city since September, mostly in Brooklyn’s Williamsburg neighborhood. The order covers four Zip codes there, Mayor Bill de Blasio (D) said Tuesday. The mandate orders all unvaccinated people in the area, including a concentration of Orthodox Jews, to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined up to $1,000."
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "doc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<span class=\"tex2jax_ignore\"><div class=\"entities\" style=\"line-height: 2.5; direction: ltr\">\n",
       "<mark class=\"entity\" style=\"background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    New York City\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">GPE</span>\n",
       "</mark>\n",
       " on \n",
       "<mark class=\"entity\" style=\"background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    Tuesday\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">DATE</span>\n",
       "</mark>\n",
       " declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases. \n",
       "<mark class=\"entity\" style=\"background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    At least 285\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">CARDINAL</span>\n",
       "</mark>\n",
       " people have contracted measles in the city since \n",
       "<mark class=\"entity\" style=\"background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    September\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">DATE</span>\n",
       "</mark>\n",
       ", mostly in \n",
       "<mark class=\"entity\" style=\"background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    Brooklyn\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">GPE</span>\n",
       "</mark>\n",
       "’s \n",
       "<mark class=\"entity\" style=\"background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    Williamsburg\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">GPE</span>\n",
       "</mark>\n",
       " neighborhood. The order covers \n",
       "<mark class=\"entity\" style=\"background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    four\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">CARDINAL</span>\n",
       "</mark>\n",
       " Zip codes there, Mayor \n",
       "<mark class=\"entity\" style=\"background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    Bill de Blasio\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">PERSON</span>\n",
       "</mark>\n",
       " (D) said \n",
       "<mark class=\"entity\" style=\"background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    Tuesday\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">DATE</span>\n",
       "</mark>\n",
       ". The mandate orders all unvaccinated people in the area, including a concentration of \n",
       "<mark class=\"entity\" style=\"background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    Orthodox\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">NORP</span>\n",
       "</mark>\n",
       " \n",
       "<mark class=\"entity\" style=\"background: #c887fb; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    Jews\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">NORP</span>\n",
       "</mark>\n",
       ", to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined \n",
       "<mark class=\"entity\" style=\"background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;\">\n",
       "    up to $1,000\n",
       "    <span style=\"font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem\">MONEY</span>\n",
       "</mark>\n",
       ".</div></span>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "displacy.render(doc, style = 'ent')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "####                           ** PROJECT ** \n",
    "####       ** TEXT CLASSIFICATION (MOVIE REVIEW IMDB AND AMAZON DATASET) **"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_yelp = pd.read_csv('NLP(folders)/yelp_label.txt',sep='\\t',header=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Wow... Loved this place.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Crust is not good.</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Not tasty and the texture was just nasty.</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Stopped by during the late May bank holiday of...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>The selection on the menu was great and so wer...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   0  1\n",
       "0                           Wow... Loved this place.  1\n",
       "1                                 Crust is not good.  0\n",
       "2          Not tasty and the texture was just nasty.  0\n",
       "3  Stopped by during the late May bank holiday of...  1\n",
       "4  The selection on the menu was great and so wer...  1"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_yelp.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "column_name = ['Review','Sentiment']\n",
    "data_yelp.columns = column_name"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Review</th>\n",
       "      <th>Sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Wow... Loved this place.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Crust is not good.</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Not tasty and the texture was just nasty.</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Stopped by during the late May bank holiday of...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>The selection on the menu was great and so wer...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              Review  Sentiment\n",
       "0                           Wow... Loved this place.          1\n",
       "1                                 Crust is not good.          0\n",
       "2          Not tasty and the texture was just nasty.          0\n",
       "3  Stopped by during the late May bank holiday of...          1\n",
       "4  The selection on the menu was great and so wer...          1"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_yelp.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1000, 2)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_yelp.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Review', 'Sentiment'], dtype='object')"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_yelp.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### LOAD AMAZON DATASET"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_amazon = pd.read_csv('NLP(folders)/amazon_labeled.txt',sep='\\t',header=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>So there is no way for me to plug it in here i...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Good case, Excellent value.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Great for the jawbone.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Tied to charger for conversations lasting more...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>The mic is great.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   0  1\n",
       "0  So there is no way for me to plug it in here i...  0\n",
       "1                        Good case, Excellent value.  1\n",
       "2                             Great for the jawbone.  1\n",
       "3  Tied to charger for conversations lasting more...  0\n",
       "4                                  The mic is great.  1"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_amazon.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_amazon.columns = column_name"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Review</th>\n",
       "      <th>Sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>So there is no way for me to plug it in here i...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Good case, Excellent value.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Great for the jawbone.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Tied to charger for conversations lasting more...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>The mic is great.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              Review  Sentiment\n",
       "0  So there is no way for me to plug it in here i...          0\n",
       "1                        Good case, Excellent value.          1\n",
       "2                             Great for the jawbone.          1\n",
       "3  Tied to charger for conversations lasting more...          0\n",
       "4                                  The mic is great.          1"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_amazon.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1000, 2)"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_amazon.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### IMDB DATA_LOAD"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_imdb = pd.read_csv('NLP(folders)/imdb.txt', sep = '\\t', header = None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>A very, very, very slow-moving, aimless movie ...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Not sure who was more lost - the flat characte...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Attempting artiness with black &amp; white and cle...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Very little music or anything to speak of.</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>The best scene in the movie was when Gerardo i...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   0  1\n",
       "0  A very, very, very slow-moving, aimless movie ...  0\n",
       "1  Not sure who was more lost - the flat characte...  0\n",
       "2  Attempting artiness with black & white and cle...  0\n",
       "3       Very little music or anything to speak of.    0\n",
       "4  The best scene in the movie was when Gerardo i...  1"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_imdb.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_imdb.columns = column_name"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(748, 2)"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_imdb.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Review</th>\n",
       "      <th>Sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>A very, very, very slow-moving, aimless movie ...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Not sure who was more lost - the flat characte...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Attempting artiness with black &amp; white and cle...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Very little music or anything to speak of.</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>The best scene in the movie was when Gerardo i...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              Review  Sentiment\n",
       "0  A very, very, very slow-moving, aimless movie ...          0\n",
       "1  Not sure who was more lost - the flat characte...          0\n",
       "2  Attempting artiness with black & white and cle...          0\n",
       "3       Very little music or anything to speak of.            0\n",
       "4  The best scene in the movie was when Gerardo i...          1"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_imdb.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1000, 2)"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_yelp.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(748, 2)"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_imdb.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1000, 2)"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_amazon.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### We have 'three' type of data now\n",
    "-  Now append all the data in a single format"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = data_yelp.append([data_imdb,data_amazon],ignore_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2748, 2)"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Review</th>\n",
       "      <th>Sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Wow... Loved this place.</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Crust is not good.</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Not tasty and the texture was just nasty.</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Stopped by during the late May bank holiday of...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>The selection on the menu was great and so wer...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              Review  Sentiment\n",
       "0                           Wow... Loved this place.          1\n",
       "1                                 Crust is not good.          0\n",
       "2          Not tasty and the texture was just nasty.          0\n",
       "3  Stopped by during the late May bank holiday of...          1\n",
       "4  The selection on the menu was great and so wer...          1"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Lets check how many positive and negative values in the data\n",
    "- 1 --> 1386 (+ve reviews)\n",
    "- 0 --> 1362 (-ve reviews)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    1386\n",
       "0    1362\n",
       "Name: Sentiment, dtype: int64"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Sentiment'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Review       0\n",
       "Sentiment    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## check the null values\n",
    "data.isnull().sum()\n",
    "\n",
    "# o/p : no null values in the data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### TOKENIZATION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "import string"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "pun = string.punctuation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'!\"#$%&\\'()*+,-./:;<=>?@[\\\\]^_`{|}~'"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pun"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- LOWE THE PUNCTUATION \n",
    "- REMOVE STOP WORDS AND NON-PUNCTUATION WORDS TO THE ARRAY\n",
    "- FINALLY PRINT THE RESULT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "def text_cleaning_data(sentence):\n",
    "    doc = nlp(sentence)\n",
    "    # LEMMATIZATION\n",
    "    tokens = []\n",
    "    for token in doc:\n",
    "        if token.lemma_!= \"-PRON-\":\n",
    "            temp = token.lemma_.lower().strip()\n",
    "        else:\n",
    "            temp = token.lower_\n",
    "        tokens.append(temp)\n",
    "    # STOP WORDS\n",
    "    cleaned_tokens = []\n",
    "    for token in tokens:\n",
    "        if token not in stopwords and token not in pun:\n",
    "            cleaned_tokens.append(token)\n",
    "    return cleaned_tokens"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['hello', 'guy', 'like', 'presentation']"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "text_cleaning_data(\"         Hello How are you guys, please like my presentation\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### VECTORIZATION ENGINEERING(TF-IDF)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import LinearSVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf = TfidfVectorizer(tokenizer = text_cleaning_data)\n",
    "classifier = LinearSVC()\n",
    "#it willm be done according to function(text_cleaning_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = data['Review']\n",
    "y = data['Sentiment']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2748,)"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2748,)"
      ]
     },
     "execution_count": 87,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2198,)"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(550,)"
      ]
     },
     "execution_count": 90,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(2198,)"
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(550,)"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(memory=None,\n",
       "     steps=[('tfidf', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',\n",
       "        dtype=<class 'numpy.float64'>, encoding='utf-8', input='content',\n",
       "        lowercase=True, max_df=1.0, max_features=None, min_df=1,\n",
       "        ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,...ax_iter=1000,\n",
       "     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,\n",
       "     verbose=0))])"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Our model is scoring:  97.49772520473158\n"
     ]
    }
   ],
   "source": [
    "print(\"Our model is scoring: \",clf.score(X_train,y_train)*100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = clf.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1,\n",
       "       0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1,\n",
       "       1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1,\n",
       "       1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,\n",
       "       0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,\n",
       "       1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1,\n",
       "       0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1,\n",
       "       0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0,\n",
       "       1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,\n",
       "       1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1,\n",
       "       1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,\n",
       "       0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,\n",
       "       1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1,\n",
       "       1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,\n",
       "       1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,\n",
       "       1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,\n",
       "       1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1,\n",
       "       0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,\n",
       "       1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0,\n",
       "       0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0,\n",
       "       1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,\n",
       "       1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,\n",
       "       0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],\n",
       "      dtype=int64)"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[212,  67],\n",
       "       [ 61, 210]], dtype=int64)"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.78      0.76      0.77       279\n",
      "           1       0.76      0.77      0.77       271\n",
      "\n",
      "   micro avg       0.77      0.77      0.77       550\n",
      "   macro avg       0.77      0.77      0.77       550\n",
      "weighted avg       0.77      0.77      0.77       550\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0], dtype=int64)"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.predict(['Wow, this is bad'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1], dtype=int64)"
      ]
     },
     "execution_count": 102,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.predict(['Worth of watching it. Please like it'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1], dtype=int64)"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.predict(['Wow, this is amzing lesson i liked it'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0], dtype=int64)"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.predict(['i hate this movie'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0], dtype=int64)"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.predict(['i hate this kind of movies but this is an exemption but i like this a little bit'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1], dtype=int64)"
      ]
     },
     "execution_count": 109,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.predict(['fantastic movie'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
