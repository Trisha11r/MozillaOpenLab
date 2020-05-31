import pandas as pd
import re
import nltk
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

def clean(data, flag):
	# stop_words = set(stopwords.words('english'))
	ps = PorterStemmer()
	lem = WordNetLemmatizer()

	data = data.replace(np.nan, 'Unknown', regex=True)

	# if flag==1:
	# 	data['Tweet_cleaned'] = ""
	# print (data)

	for index, row in data.iterrows():
		s = ''
		row_tweet = row['Actual_Tweet'].replace('\\n', ' ')
		tweet = row_tweet.split(' ')

		urlFound = False

		for i, c in enumerate(tweet):
			if( c.startswith('b') and c.endswith('RT') ):
				# s += '_RT_ '
				continue
			elif( c.startswith('@') and c.endswith(':') ):
				# s += '_MENTION_ '
				continue
			elif ( 'http' in c ):
				# s += '_URL_ '
				urlFound = True
				
				t = ""
				for x in c:
					if ( (x>='a' and x<='z') or (x>='A' and x<='Z') or (x>='0' and x<='9') or x==':' or x=='/' or x=="."):
						t += x
					elif x=='\\':
						break

				data.at[index, 'URL'] = t
				continue
			elif ( re.search("^.*x[a-z][0-9].*$" , c) ):
				# s += ''
				continue
			elif ( c.startswith('#') or c.startswith('@')):
				if flag==1 and c.startswith('#'):
					s += c + ' '
				else:
					continue

			else:
				if( i==0 and (c.startswith('b\'') or c.startswith('b\"') ) ):
					c = c[2:]

				if flag==0:
					c = c.lower()

					# Removing Stop Words using nltk
					if c in stop_words:
						continue
				
				# # Removing Stop Words using nltk
				# if flag==0 and c in stop_words:
				# 	continue

				t = ''
				ind = -1
				count = -1

				for x in c:
					
					ind += 1
					if flag==0 and x=='#':
						break

					if ( (x>='a' and x<='z') or (x>='A' and x<='Z') ):
						t += x
						count = 0
					elif count==0:
						t += ' '
						count = -1

				if flag==0:
					t = lem.lemmatize(t, pos='v')
					if t in stop_words:
						continue

				s += t + ' '

		
		if s=='':
			print (index)

		if flag==0:
			# Removing one, two letter words
			s = re.sub(r'\b\w{1,2}\b', '', s)

		# Remove extra whitespaces in between
		s = " ".join(s.split())
		
		if flag==0:
			row['Tweet'] = s
		else:
			data.at[index, 'Tweet_cleaned'] = s
			if urlFound==False:
				data.at[index, 'URL'] = 'Not Available'
	
	if flag==0:	
		data = data.drop_duplicates(subset=['Tweet'], keep='first')
		data['Tweet'].replace('', np.nan, inplace=True)
		data.dropna(subset=['Tweet'], inplace=True)
	# else:
	# 	print (data['Tweet_cleaned'])
	# 	data = data.drop_duplicates(subset=['Tweet_cleaned'], keep='first')
	# 	data['Tweet_cleaned'].replace('', np.nan, inplace=True)
	# 	data.dropna(subset=['Tweet_cleaned'], inplace=True)
	
	print ('Data cleaned')
	# print (data)

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

pd.set_option('display.max_colwidth', -1)
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


result_csv = pd.DataFrame()
# location_results = pd.DataFrame()
last_pg = -1
location = ""

def requestResults(name):
	global result_csv

	label = loaded_models['d_nd'].predict([name])[0]

	if label==1:
		req_off = loaded_models['req_off'].predict([name])[0]
		res_type = loaded_models['res_type'].predict([name])[0]
		result_csv = data[(data['Labels']==label) & (data['Donation_Related']==req_off) & (data['Resource_Type']==res_type)]

		result_csv = result_csv[['Time', 'Tweet_cleaned', 'URL', 'Location']]
		result_csv.reset_index(drop=True, inplace=True)
		result_csv.index += 1
	else:
		result_csv = pd.DataFrame()
	
	# return result_csv

app = Flask(__name__)

@app.route("/")
def main():
	return render_template('index.html')


@app.route('/searched', methods=['POST', 'GET'])
def get_data():
	global location
	global result_csv
	# global location_results

	if request.method == 'POST':
		user = request.form['search']
		location = request.form['location']

		requestResults(user)
		print ('Searched : ', user)
		print ('result_csv (in searched): ', result_csv)

		f = open('templates/trying.html').read()
		soup = Soup(f, features="html.parser")
		p = soup.find("p", {"class" : "searched_for"})
		paginate = soup.find("div", {"class" : "pagination"})

		if result_csv.empty:
			p.append("You searched for: " + user + ". This is a Non-Donation request.")
		else:
			
			# Check for tweets at the location given. If location not given, then show results for the world (only 30 tweets per page).
			if location != "":
				show_results = result_csv[result_csv['Location'].str.contains(location.upper()) | result_csv['Location'].str.contains(location.lower())]

				location_results = show_results

				# If no tweets are present at searched location
				if len(show_results)==0:
					
					show_results = result_csv[:30]
					p.append("You searched for: " + user + " at " + location + ". Found 0 results. Displaying " + str(len(result_csv)) + " results for other locations.")
					

					location = ""
				else:
					result_csv = result_csv[~result_csv['Location'].str.contains(location.upper()) & ~result_csv['Location'].str.contains(location.lower())]
					
					# Showing only top 15 tweets of searched location
					if len(show_results) > 15:
						show_results = show_results[:15]

					p.append("You searched for: " + user + " at " + location + ". Found " + str(len(show_results)) + " results.")
			else:
				show_results = result_csv[:30]
				p.append("You searched for: " + user + " at " + location + ". Found " + str(len(result_csv)) + " results.")					
			

			show_results.reset_index(drop=True, inplace=True)
			show_results.index += 1
				
			result = Soup(show_results.to_html(), features="html.parser")

			result.find("tr")['style'] = 'text-align:center;'


			# Make URLs as hyperlinks
			count = 0
			insert = 3
			for td in result.find_all("td"):
				count += 1

				if (count==insert):

					if (td.text!="Not Available"):

						a = soup.new_tag("a")
						a["href"] = td.text
						a.string = td.text

						td.string = ""
						td.append(a)

					insert += 4

				if count == insert-2 :
					td['style'] = "width:12%;"

			table = result.find("table")
			table['border'] = '0'
			table['style'] = 'position:absolute;top:180px;padding-left:35px;padding-right:35px;text-align:center;'

			p.insert_after(result)

			a = soup.find("a", {"id" : 1})
			
			if location=="":
				n = len(result_csv)//30

				if n>20:
					paginate["style"] = "position:absolute;left:50%;top:185%;width:80%;transform: translate(-50%, -50%); background-color: #525252;background-size: cover;"
				else:
					paginate["style"] = "position:absolute;left:50%;top:185%;transform: translate(-50%, -50%); background-color: #525252;background-size: cover;"
			else:
				n = len(result_csv)//15

				if n>20:
					paginate["style"] = "position:absolute;left:50%;top:195%;width:80%;transform: translate(-50%, -50%); background-color: #525252;background-size: cover;"
				else:
					paginate["style"] = "position:absolute;left:50%;top:195%;transform: translate(-50%, -50%); background-color: #525252;background-size: cover;"

			if (n%30 != 0):
				n += 1
				last_pg = n

			for i in range(n-1):
				pages = soup.new_tag("a")
				pages['class'] = 'inactive'
				pages['id'] = i+2
				pages['onclick'] = "redirectPage(this.id)"
				pages.string = str(i+2)
				a.insert_after(pages)
				a = pages

			if location != "":
				# Other Location Results
				p = soup.new_tag("p")
				p['class'] = "other_results_para"
				p['style'] = "position: absolute;top:750px;font-weight: bold;"
				p.string = "Other Location Tweets (Found " + str(len(result_csv)) + " results)"

				table = soup.find("table", {"class" : "dataframe"})
				table.insert_after(p)

				other_results = result_csv[:15]
				other_results.reset_index(drop=True, inplace=True)
				other_results.index += 1
					
				result = Soup(other_results.to_html(), features="html.parser")

				result.find("tr")['style'] = 'text-align:center;'


				# Make URLs as hyperlinks
				count = 0
				insert = 3
				for td in result.find_all("td"):
					count += 1

					if (count==insert):

						if (td.text!="Not Available"):

							a = soup.new_tag("a")
							a["href"] = td.text
							a.string = td.text

							td.string = ""
							td.append(a)

						insert += 4

					if count == insert-2 :
						td['style'] = "width:12%;"

				table = result.find("table")
				table['class'] = 'other_results'
				table['border'] = '0'
				table['style'] = 'position:absolute;top:800px;padding-left:35px;padding-right:35px;text-align:center;'

				p = soup.find("p", {"class" : "other_results_para"})
				p.insert_after(table)


		file = open('templates/searched.html', "w", encoding="utf-8")
		file.write(str(soup))
		file.close()

		return render_template('searched.html')

@app.route('/page')
def pagination():
	global result_csv
	# global location_results
	global last_pg
	global location

	print ('result_csv (in page): ', result_csv)
	# Get the current page as the argument in URL
	pg = request.args.get('page', default = "1", type = str)

	# Parse the searched.html file for updating the new table
	f = open('templates/searched.html', encoding='utf-8').read()
	soup = Soup(f, features="html.parser")

	p = soup.find("p", {"class" : "searched_for"})

	# If location is empty, then delete previous table results
	if location=="":
		# Remove the table tag for previous page results from the html file.
		for s in soup.select('table'):
			s.extract()

		tweets_per_pg = 30
		
	# Otherwise, delete other results table and display data for next page
	else:
		for s in soup.find_all("table", {"class" : "other_results"}):
			s.decompose()

		tweets_per_pg = 15

	# Make the previous page class as inactive
	a = soup.find("a", {"class" : "active"})
	a["class"] = "inactive" 
	
	# Make the current page (pg) class as active
	a = soup.find("a", {"id" : pg})
	a['class'] = "active"

	pg = int(pg)

	if(pg==last_pg):
		# Get remaining results from result_csv
		show_results = result_csv.loc[(pg-1)*tweets_per_pg + 1:]
	else:
		# Get only 30 results from result_csv depending upon the page number
		show_results = result_csv.loc[(pg-1)*tweets_per_pg + 1: pg * tweets_per_pg]
	
	print ('page: ', pg)
	print ('show_results (in page): ', show_results)

	result = Soup(show_results.to_html(), features="html.parser")
	result.find("tr")['style'] = 'text-align:center;'
	# Make URLs as hyperlinks
	count = 0
	insert = 3
	for td in result.find_all("td"):
		count += 1

		if (count==insert):

			if (td.text!="Not Available"):

				a = soup.new_tag("a")
				a["href"] = td.text
				a.string = td.text

				td.string = ""
				td.append(a)

			insert += 4
			
		if count == insert-2 :
			td['style'] = "width:12%;"

	table = result.find("table")
	table['border'] = '0'

	if location != "":
		table["class"] = "other_results"
		table['style'] = 'position:absolute;top:800px;padding-left:35px;padding-right:35px;text-align:center;'
		p = soup.find("p", {"class" : "other_results_para"})
		
	else:
		table['style'] = 'position:absolute;top:180px;padding-left:35px;padding-right:35px;text-align:center;'

	p.insert_after(table)
	

	file = open('templates/searched.html', "w", encoding="utf-8")
	file.write(str(soup))
	file.close()

	return render_template('searched.html')	


if __name__ == '__main__':
	
	app.run(debug=True, use_reloader=True)