import pandas as pd
import re
import nltk
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import pickle
from joblib import load

from bs4 import BeautifulSoup as Soup
import html2text

import sklearn
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics
from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
					 "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
					 "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
					 "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
					 "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
					 "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
					 "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more"
					 , "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
					 "ought", "our", "ours    ourselves", "out", "over", "own", "same", "shan't", "she",
					 "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
					 "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
					 "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
					 "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
					 "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
					 "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
					 "your", "yours", "yourself", "yourselves", 'a', 'about', 'above', 'across', 'after', 'again',
					 'against', 'all', 'almost', 'alone', 'btw', 'north', 'south', 'east', 'west', 'sarita', 'woke', 'wake',
					 'suv', 'omg', 'asap', 'contain', 'au', 'demi', 'mam', 'sir', "ma'am", "i'm'", 'ohh', 'oh', 'duh',
					 'go', 'goes', 'went', 'gone', 'dollar', 'dollars', 'cents', 'cent', 'usa', 'dont', 'aaa',
					 'along', 'already', 'also', 'although', 'always', 'among', 'an', 'and', 'another', 'any',
					 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask',
					 'asked', 'asking', 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be',
					 'became', 'because', 'become', 'becomes', 'been', 'before', 'began', 'behind', 'being',
					 'beings', 'between', 'both', 'but', 'by', 'c', 'came', 'can', 'cannot', 'couldnt',
					 'case', 'cases', 'certain', 'certainly', 'clear', 'clearly', 'come', 'could', 'coz', 'd', 'did',
					 'differ', 'different', 'differently', 'do', 'does', 'done', 'down', 'downed', 'downing', 'downs',
					 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even',
					 'evenly', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'f',
					 'faces', 'fact', 'facts', 'far', 'felt', 'few', 'find', 'first', 'for', 'four', 'from',
					 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general',
					 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'going', 'goods', 'got',
					 'group', 'grouped', 'grouping', 'groups', 'h', 'had', 'has',
					 'have', 'having', 'he', 'her', 'here', 'herself',
					 'him', 'himself', 'his', 'how', 'however', 'i', 'if', 'important', 'in', 'into', 'is', 'it', 'its',
					 'itself', 'j', 'just', 'k', 'keep', 'keeps',
					 'knew', 'know', 'known', 'knows', 'l', 'largely', 'later', 'latest',
					 'least', 'let', 'lets', 'likely', 'm', 'made', 'make',
					 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more', 'most',
					 'mostly', 'mr', 'mrs', 'much', 'must', 'my', 'myself', 'n', 'necessary', 'need', 'needed',
					 'needing', 'needs', 'new', 'next',
					 'noone', 'nothing', 'now', 'nowhere', 'number', 'numbers', 'o', 'of', 'off', 'often',
					 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening', 'opens', 'or',
					 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'our', 'out', 'over', 'p', 'part',
					 'parted', 'parting', 'parts', 'per', 'perhaps', 'place', 'places', 'point', 'pointed', 'pointing',
					 'points', 'possible', 'present', 'presented', 'presenting', 'presents',
					 'put', 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'room', 'rooms', 's',
					 'said', 'same', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming',
					 'seems', 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'side',
					 'sides', 'since', 'so', 'some', 'somebody', 'someone', 'something',
					 'somewhere', 'state', 'states', 'still', 'such', 'sure', 't', 'take', 'taken', 'than',
					 'that', 'the', 'their', 'them', 'then', 'there', 'therefore', 'these', 'they', 'thing', 'things',
					 'think', 'thinks', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus',
					 'to', 'today', 'together', 'too', 'took', 'toward', 'turn', 'turned', 'turning', 'turns', 'two',
					 'u', 'under', 'until', 'up', 'upon', 'us', 'use', 'used', 'uses', 'v', 'very', 'w', 'want',
					 'wanted', 'wanting', 'wants', 'was', 'way', 'ways', 'we', 'We', 'well', 'wells', 'went', 'were', 'what',
					 'when', 'where', 'whether', 'which', 'while', 'who', 'whole', 'whose', 'why', 'will', 'with',
					 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year', 'years',
					 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours', 'z', 'weren', 'didn', 'ours', 'hasn', 'hadn', "should've", 'ourselves', 're', 'wouldn', 've', 'ain', 'couldn', 'mustn', 'aren', 'isn', 'wasn', 'doesn', 'll', "that'll", 'mightn', 'won', 'shan', "mightn't", "needn't", 'haven', 'needn', 'ma', 'don', 'shouldn']

def clean(data):
	# stop_words = set(stopwords.words('english'))
	ps = PorterStemmer()
	lem = WordNetLemmatizer()

	data = data.replace(np.nan, 'Unknown', regex=True)

	for index, row in data.iterrows():
		s = ''
		row_tweet = row['Actual_Tweet'].replace('\\n', ' ')
		tweet = row_tweet.split(' ')
		for i, c in enumerate(tweet):
			if( c.startswith('b') and c.endswith('RT') ):
				# s += '_RT_ '
				continue
			elif( c.startswith('@') and c.endswith(':') ):
				# s += '_MENTION_ '
				continue
			elif ( 'http' in c ):
				# s += '_URL_ '
				continue
			elif ( re.search("^.*x[a-z][0-9].*$" , c) ):
				# s += ''
				continue
			elif ( c.startswith('#') or c.startswith('@')):
				continue
			else:
				if( i==0 and (c.startswith('b\'') or c.startswith('b\"') ) ):
					c = c[2:]

				c = c.lower()
				# Removing Stop Words using nltk
				if c in stop_words:
					continue

				t = ''
				ind = -1
				count = -1

				for x in c:
					
					ind += 1
					if x=='#':
						break

					if ( (x>='a' and x<='z') or (x>='A' and x<='Z') ):
						t += x
						count = 0
					elif count==0:
						t += ' '
						count = -1

				t = lem.lemmatize(t, pos='v')
				if t in stop_words:
					continue
				s += t + ' '

		
		if s=='':
			print (index)

		# Removing one, two letter words
		s = re.sub(r'\b\w{1,2}\b', '', s)

		# Remove extra whitespaces in between
		s = " ".join(s.split())


		row['Tweet'] = s
	
	data = data.drop_duplicates(subset=['Tweet'], keep='first')
	data['Tweet'].replace('', np.nan, inplace=True)
	data.dropna(subset=['Tweet'], inplace=True)
	
	print ('Data cleaned')


	return data

def labelling(data):

	# data = pd.read_csv('TwitterDataCleaned.csv')

	search_words = ['donate', 'donation', 'aid', 'request', 'offer', 'relief', 'handout', 'assistance', 'charity', 'contribution', 'help', 'endowment', 'offering', 'grant', 'contribute', 'support', 'money', 'monetary','assist', 'fund', 'service', 'benefaction']

	indices = []

	for index, row in data.iterrows():
		print (index, row['Tweet'])
		for word in search_words:
			if( re.search("^.*" + word +".*$" , row['Tweet'] )):
				# print ('word: ', word, ' Tweet: ', row['Tweet'])
				indices.append(index)
				break
	print (len(indices))
	data['Labels'] = ['ND'] * len(data)

	for index, row in data.iterrows():
		if index in indices:
			data['Labels'][index] = 'D'

	unknown = data[data['Labels']=='ND']
	donation = data[data['Labels']=='D']
	
	# data.to_csv('TwitterDataLabelled.csv', index=False)	
	return donation, unknown, data

def split(data):
	data = np.array_split(data, 4)

	data[0].to_csv('division/Kartik.csv', index=False)
	data[1].to_csv('division/Trisha.csv', index=False)
	data[2].to_csv('division/Nitin.csv', index=False)
	data[3].to_csv('division/Shanuj.csv', index=False)

def label_encoder(data):

	data = data.reset_index(drop=True)
	
	data.loc[data['Labels']==0, ['Donation_Related', 'Resource_Type']] = 'N/A'
	
	encoder = {'ND':0, 'D':1, 'R':0, 'O':1, 'Money':0, 'Clothing':1, 'Food':2, 'Medical':3, 'Shelter':4, 'Volunteer':5, 'N/A':-1}

	# data['Labels'] = data['Labels'].map({'ND': 0, 'D': 1})
	data['Donation_Related'] = data['Donation_Related'].map({'R':0, 'O':1, 'N/A':-1})
	data['Resource_Type'] = data['Resource_Type'].map({'Money':0, 'Clothing':1, 'Food':2, 'Medical':3, 'Shelter':4, 'Volunteer':5, 'N/A':-1})

	print (len(data[data['Labels']==0]))
	print (len(data[data['Labels']==1]))

	data.to_csv('dataset_final_3k.csv', index=False)

def loadModel():

	classes = ['Labels', 'Donation_Related', 'Resource_Type']
	
	for cls in classes:
		
		if cls=='Labels':
			print ('Loading Model for Donation/Non-Donation...')
		  
			data = pd.read_csv('training_data.csv')
			# load the saved pipleine model
			loaded_models['d_nd'] = load("models/donation_nonDonation.joblib")

		elif cls=='Donation_Related':
			print ('Loading Model for Only Donation Data- Request/Offer...')
		  
			data = pd.read_csv('training_data_donation.csv')
			loaded_models['req_off'] = load("models/request_offer.joblib")
		else:
			print ('Loading Model for Only Donation Data- Resource Type...')
		  
			data = pd.read_csv('training_data_donation.csv')
			loaded_models['res_type'] = load("models/resource_type.joblib")
		
		sentence = ['Want to donate.']
		print (loaded_models['d_nd'].predict(sentence))
		# break

	return "test"

data = pd.read_csv('final_dataset.csv')

loaded_models = {}
classes = ['Labels', 'Donation_Related', 'Resource_Type']

for cls in classes:
	
	if cls=='Labels':
		print ('Loading Model for Donation/Non-Donation...')

		loaded_models['d_nd'] = load("models/donation_nonDonation.joblib")

	elif cls=='Donation_Related':
		print ('Loading Model for Only Donation Data- Request/Offer...')

		loaded_models['req_off'] = load("models/request_offer.joblib")
	else:
		print ('Loading Model for Only Donation Data- Resource Type...')

		loaded_models['res_type'] = load("models/resource_type.joblib")

def requestResults(name):
	label = loaded_models['d_nd'].predict([name])[0]

	if label==1:
		req_off = loaded_models['req_off'].predict([name])[0]
		res_type = loaded_models['res_type'].predict([name])[0]
		result_csv = data[(data['Labels']==label) & (data['Donation_Related']==req_off) & (data['Resource_Type']==res_type)]

		result_csv = result_csv[['Time', 'Actual_Tweet', 'Location']]
		result_csv.reset_index(drop=True, inplace=True)
	else:
		result_csv = pd.DataFrame()
	
	return result_csv

app = Flask(__name__)

@app.route("/")
def main():
	print ()
	print ()
	print ('in main')
	return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
	print ()
	print ()
	print ("in get_data")
	if request.method == 'POST':
		user = request.form['search']
		result_csv = requestResults(user)
		
		f = open('templates/trying.html').read()
		soup = Soup(f, features="html.parser")
		p = soup.find("p", {"class" : "searched_for"})

		if result_csv.empty:
			p.append("You searched for: " + user + ". This is a Non-Donation request.")
		else:
			result_csv = result_csv[:31]
			p.append("You searched for: " + user + ". Found " + str(len(result_csv)) + " results.")
			result = Soup(result_csv.to_html(), features="html.parser")

			table = result.find("table")
			table['style'] = 'position:absolute;top:180px;padding:35px;'

			p.insert_after(result)
			

		file = open('templates/searched.html', "w", encoding="utf-8")
		file.write(str(soup))
		file.close()

		return render_template('searched.html')


if __name__ == '__main__':
	
	app.run(debug=True, use_reloader=True)