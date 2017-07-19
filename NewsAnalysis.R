# Kelly Xie
# Prof. Tambe, ADE Spring 2017
# Homework 2: News Analytics


########### 1. Fit a topic model ##############


# imports archive data
news = read.csv("/data/newsarticledata.csv")

# cleans and reprocesses the text
library("tm")
corp.original = VCorpus(VectorSource(news$content))
corp = tm_map(corp.original, removePunctuation)
corp = tm_map(corp, removeNumbers)
corp = tm_map(corp, content_transformer(removeWords), stopwords("SMART"),lazy=TRUE)
corp = tm_map(corp, content_transformer(tolower),lazy=TRUE)
corp = tm_map(corp, content_transformer(stemDocument),lazy=TRUE)
corp = tm_map(corp, stripWhitespace)

# generates a document-term matrix
dtm = DocumentTermMatrix(corp)
dtms = removeSparseTerms(dtm, .995)
dtm_matrix = as.matrix(dtms)

# trains a topic model
library(topicmodels)
terms = rowSums(dtm_matrix) != 0 #sets parameters
dtm_matrix = dtm_matrix[terms,]
lda_result = LDA(dtm_matrix, 10, method="Gibbs") #fits LDA model
LDA = terms(lda_result, 20)

# assigns names to categories
colnames(LDA) = c("Real Estate", "Careers", "Finance",
                  "Government", "Economy", "Healtcare", "Politics",
                  "Earnings", "Energy", "Social Media")
print(LDA)


########### 2. Retrieve new articles ##############


# extracts article URLs and text
library(rvest)
library(stringr)

article.urls=NULL
# loop for scraping all article URLs from the first 10 pages
for (i in (1:10)) {
  urls = read_html(paste("http://www.cnbc.com/us-news/?page=", i, sep="")) %>%
    html_nodes(".bigHeader .headline a") %>%
    html_attr("href")
  article.urls = cbind(article.urls, urls)
}
article.urls = article.urls[-(24:26)] # removes non-text articles/videos
head(article.urls, 40) # URLS of first 40 articles

article.text=NULL
# loop for collecting and cleaning the textual data from each article
for (i in 1:length(article.urls)) {
  text = read_html(paste("http://www.cnbc.com", article.urls[i], sep="")) %>%
    html_nodes("p") %>%
    html_text()
  text = text[-1] # removes first element in vector
  text = gsub("^\\s+|\\s+$", "", text, fixed = FALSE) # strips leading and trailing whitespace
  text = gsub('"', '', text, fixed = FALSE) # replaces all quotations
  text = paste(text, collapse=" ") # compresses text into one block
  article.text = cbind(article.text, text)
}
length(article.text) # total number of articles
article.text[1:4] # content of first 4 articles


########### 3. Classify news articles using topic model ##############


# creates a new document term matrix
news.corp = VCorpus(VectorSource(article.text))

# specifies dictionary when creating the dtm for the new articles,
# which will limit the dtm it creates to only the words that also appeared in the archive
dic = Terms(dtms)
new_dtm = DocumentTermMatrix(news.corp, control=list(dictionary = dic))
new_dtm = new_dtm[rowSums(as.matrix(new_dtm))!=0,]
topic_probabilities = posterior(lda_result, new_dtm)

# renames columns as categories
colnames(topic_probabilities$topics) = colnames(LDA)
print(topic_probabilities$topics)

# generates a vector that assigns to each document
# the topic for which it has the highest probability of appearing
categories=NULL
for (i in 1:length(article.urls)) {
  max = which.max(topic_probabilities$topics[i,])
  cat = colnames(topic_probabilities$topics)[max]
  categories = rbind(categories, cat)
}
colnames(categories) = "Category" # renames column
rownames(categories) = c(1:length(article.urls)) # number documents/articles
print(categories)

# in a table, prints the contents of any ten news articles 
# and the corresponding categories
news.content=NULL
news.category=NULL
random_news = sample(1:length(article.urls), 10, replace=F) # random article generator
for (i in random_news) {
  # get contents of selected articles
  news.content = rbind(news.content, article.text[i])
  # get categories of selected articles
  news.category = rbind(news.category, categories[i])
}

# combines contents and categories into one table
news.table = cbind(news.category, news.content)
news.table = as.data.frame(news.table)
colnames(news.table) = c("Category", "Content")
View(news.table)

