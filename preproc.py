import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from sklearn.semi_supervised import LabelPropagation
from sklearn import manifold
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pandas.io.json import json_normalize

def load_dataset():
	sources = ('BuzzFeed','PolitiFact')
	news_types = ('Fake','Real')
	direct_format = './FakeNewsNet/Data/{}/{}NewsContent/{}_{}_{}-Webpage.json'
	labels = []
	true_labels = []
	fake_labels = []
	raw_data = []
	i = 0
	fields = set()

	for source in sources:
		for news_type in news_types:
			max_range = 0
			if(source == 'BuzzFeed'):
				max_range = 92
			else:
				max_range = 121
			for news_num in range(1, max_range):
				if(news_type == 'Fake'):
					labels.append(0)
					fake_labels.append(i)
				else:
					labels.append(1)
					true_labels.append(i)

				#print('colocando noticia num {} da do tipo {} da fonte {}'.format(str(news_num), news_type, source))
				news_directory = direct_format.format(source, news_type, source, news_type, str(news_num))

				with open(news_directory) as f:
					raw_json = json.load(f)#carrega o json como dict
					raw_data.append(raw_json)
				i += 1

	return (raw_data, labels)

def df_from_raw_data(raw_data, labels):
	news_dict = []
	incomplete_news_list = []

	news_dataframe = pd.DataFrame()
	i = 0
	for news in raw_data:

		dict_entry = {}
			
		if('meta_data' in news.keys() and 'og' in news['meta_data'].keys()):
			dict_entry['title'] = news['meta_data']['og']['title']
		else:
			dict_entry['title'] = news['title']
			incomplete_news_list.append(news)

		dict_entry['text'] = news['text']
		
		#print('colocando class {} na noticia de index {} titulo {}'.format(labels[i], i, dict_entry['title']))
		dict_entry['class'] = labels[i]
		
		news_series = pd.Series(dict_entry)
		#print(news_series)
		news_dataframe = news_dataframe.append(news_series, ignore_index=True)
		news_dict.append(dict_entry)
		i += 1
		
	#print('nro noticias incompletas: ', len(incomplete_news_list))
	#print('noticias inseridas',len(news_dict))
	#print('DATAFRAME GERADO')
	#print(news_dataframe.columns)

	return news_dataframe

def df_get_named_entities(dataframe):
	return 0

def df_to_tfidf(dataframe, column):
	vectorizer = TfidfVectorizer(stop_words='english')
	bag_of_words_matrix = vectorizer.fit_transform(dataframe[column])

	#print(type(bag_of_words_matrix))
	#print('vocabulario do vectorizer ', column,' ', vectorizer.vocabulary_)
	#print('numero de palavras da coluna {}: {}'.format(column, max(vectorizer.vocabulary_[word] for word in vectorizer.vocabulary_.keys())))
	return bag_of_words_matrix

def df_to_file(dataframe, name):
	pd.set_option('display.max_colwidth', -1)
	name = name+'.csv'
	dataframe.to_csv(name)
	return 0

def plot_graph(dataframe):
	graph = nx.from_numpy_matrix(dataframe, create_using=nx.Graph())
	nx.draw(graph, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
	plt.show()
	return 0
	
def pick_labels_semisupervised(known_labels, fake_labels, true_labels, test_label_qtt):
	test_labels = np.full(shape=len(known_labels), fill_value=-1)
	fake_label_qtt = 0
	true_label_qtt = 0
	while(fake_label_qtt < int(test_label_qtt/2)):
		chosen_label = np.random.randint(low=0, high=len(fake_labels))
		test_labels[chosen_label] = 0
		fake_label_qtt += 1
	
	while(true_label_qtt < int(test_label_qtt/2)):
		chosen_label = np.random.randint(low=0, high=len(true_labels))
		test_labels[chosen_label] = 1
		true_label_qtt += 1
	
	return test_labels

def get_percentile_cm(cm):
	cm_sum = sum(sum(cm))
	cm = np.multiply(np.divide(cm, cm_sum),100)
	return cm

def lb_prop_classify(network, labels):
	kf = StratifiedKFold(n_splits=10)
	scores = []
	cms = []

	for test_index, train_index in kf.split(network ,labels):
		first_train_index, last_train_index = min(train_index), max(train_index)

		train_dataset = network[first_train_index:last_train_index]
		train_labels = labels[first_train_index:last_train_index]

		test_dataset = np.delete(network, np.s_[first_train_index:last_train_index], 0)
		test_labels = np.delete(labels, np.s_[first_train_index:last_train_index], 0)

		label_spreading_model = LabelPropagation()
		label_spreading_model.fit(train_dataset, train_labels)
		scores.append(label_spreading_model.score(test_dataset, test_labels))

		prediction = label_spreading_model.predict(test_dataset)
		cms.append(confusion_matrix(test_labels, prediction, label_spreading_model.classes_))

	print('label propagation media {}'.format(np.average(scores)))
	print('label propagation desvio padrao {}'.format(np.std(scores)))
	print('label propagation matriz de confusao')
	print(get_percentile_cm(get_average_cm(cms)))
	print('\n')

	return scores

def network_to_es_df(network, labels):
	embedding = manifold.locally_linear_embedding(X=network, n_neighbors=5, n_components=2)
	embedded_x = embedding[0][:,0]
	embedded_y = embedding[0][:,1]
	embedded_df = pd.DataFrame()
	embedded_df['x'] = embedded_x
	embedded_df['y'] = embedded_y
	class_df = pd.DataFrame() 
	class_df['class'] = labels
	return embedded_df, class_df

def embedded_df_classify(embedded_df, class_df):
	score = 0
	kf = StratifiedKFold(n_splits=10)
	n_splits = kf.get_n_splits(embedded_df, class_df)
	rf_scores = []
	knn_scores = []
	lr_scores = []

	rf_cms = []
	knn_cms = []
	lr_cms = []

	rf_classifier = RandomForestClassifier(criterion='entropy')
	knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
	lr_classifier = LogisticRegression()

	for train_index, test_index in kf.split(embedded_df, class_df):
		#print('train index {} test_index {}'.format(train_index, test_index))
		first_test_index, last_test_index = min(test_index), max(test_index)
		#print('\nvai remover de {} ate {} do dataset de treino'.format(first_test_index, last_test_index))
		train_dataset = embedded_df.drop(embedded_df.index[first_test_index:last_test_index], axis=0)
		test_dataset = embedded_df.iloc[first_test_index:last_test_index]
		train_labels = class_df.drop(class_df.index[first_test_index:last_test_index], axis=0)
		test_labels = class_df.iloc[first_test_index:last_test_index]

		rf_classifier.fit(train_dataset, np.ravel(train_labels))
		rf_scores.append(rf_classifier.score(test_dataset, test_labels))

		knn_classifier.fit(train_dataset, np.ravel(train_labels))
		knn_scores.append(knn_classifier.score(test_dataset, test_labels))

		lr_classifier.fit(train_dataset, np.ravel(train_labels))
		lr_scores.append(lr_classifier.score(test_dataset, test_labels))

		prediction = rf_classifier.predict(test_dataset)
		rf_cms.append(confusion_matrix(test_labels, prediction, rf_classifier.classes_))
		
		prediction = knn_classifier.predict(test_dataset)
		knn_cms.append(confusion_matrix(test_labels, prediction, knn_classifier.classes_))

		predcition = lr_classifier.predict(test_dataset)
		lr_cms.append(confusion_matrix(test_labels, prediction, lr_classifier.classes_))

	print('random forest media: {}'.format(np.average(rf_scores)))
	print('random forest desvio padrao: {}'.format(np.std(rf_scores)))
	print('random forest matriz de confusao')
	print(get_percentile_cm(get_average_cm(rf_cms)))
	print('\n')

	print('knn media {}'.format(np.average(knn_scores)))
	print('knn desvio padrao: {}'.format(np.std(knn_scores)))
	print('knn matriz de confusao')
	print(get_percentile_cm(get_average_cm(knn_cms)))
	print('\n')

	print('regressao logistica media {}'.format(np.average(lr_scores)))
	print('regressao logistica desvio padrao {}'.format(np.std(lr_scores)))

	return rf_scores

def get_average_cm(cms):
	average_cm = np.zeros((2,2))
	total_tp = 0
	total_fp = 0
	total_fn = 0
	total_tn = 0
	i = 0
	for cm in cms:
		total_tp += cm[0][0]
		total_fp += cm[0][1]
		total_fn += cm[1][0]
		total_tn += cm[1][1]
		i += 1
	average_cm[0][0] = total_tp/len(cms)
	average_cm[0][1] = total_fp/len(cms)
	average_cm[1][0] = total_fn/len(cms)
	average_cm[1][1] = total_tn/len(cms)

	return average_cm

def raw_data_to_df(raw_data, labels):
	df = pd.DataFrame()
	i = 0
	for data in raw_data:
		data['label'] = labels[i]
		df = df.append(json_normalize(data), sort=True)
		i += 1
	return df

def run():
	#pd.set_option('display.height', 1000)
	pd.set_option('display.max_rows', 500)
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)
	file_data = load_dataset()
	raw_data = file_data[0]
	labels = file_data[1]

#	print('ml dict keys', raw_data[0].keys())
	
#	flat_dict = json_normalize(raw_data[0])
	dataframe = raw_data_to_df(raw_data, labels)
	print('nro linhas {} nro colunas {}'.format(dataframe.shape[0], dataframe.shape[1]))

	#print(df['authors'])
	#print('df shape',df.shape)
	#print(labels)
	#print(df['label'])
	#print(df.dtypes)
    
	#for data in raw_data:
	#	flattened_data = json_normalize(data)
	#	df= df.append(flattened_data)

#	print('flattened json', flat_dict)
#	print('\n\n\ntipo do json normalizado', type(flat_dict))
#	flat_dict.to_json('flat_json.txt')
#	with open('ml_json1.txt','w') as f:
#		f.write(json.dumps(raw_data[0]))

#	flat_dict2 = json_normalize(raw_data[1])
#	print('SECOND FLATTENED JSON', flat_dict2)
#	print(flat_dict.to_json('flat_json2.txt'))
#	with open('ml_json2.txt','w') as f:
#		f.write(json.dumps(raw_data[1]))

	#fake_labels = file_data[2]
	#true_labels = file_data[3]

	#print('numero de textos',len(raw_data))
	#print("% verdadeiros {}--- % fake {}\n".format(100*len(true_labels)/len(labels), 100*len(fake_labels)/len(labels)))

	#dataframe = df_from_raw_data(raw_data, labels)
	news_body_tfidf = df_to_tfidf(dataframe, 'text')
	news_body_network = cosine_similarity(news_body_tfidf)

	#plot_graph(network)

	print('CLASSIFICACOES USANDO CORPO DA NOTICIA')
	lb_prop_classify(news_body_network, labels)
	
	#embedded_data = network_to_es_df(news_body_network, labels)
	#embedded_df = embedded_data[0]
	#class_df = embedded_data[1]
	#embedded_df_classify(embedded_df, class_df)

	#print('\nCLASSIFICACOES USANDO TITULO DA NOTICIA')
	#title_tfidf = df_to_tfidf(dataframe, 'title')
	#title_network = cosine_similarity(title_tfidf)
	#lb_prop_classify(title_network, labels)
	#embedded_data = network_to_es_df(title_network, labels)
	#embedded_df = embedded_data[0]
	#class_df = embedded_data[1]
	#embedded_df_classify(embedded_df, class_df)

#run()