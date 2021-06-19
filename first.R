##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Lets see edx
dim(edx)
head(edx)

#It has more than 9 million samples and columns like title timestamp (of rating),
#userId (the ones that rates) ,movieId and title (of rated movie), genres (category)
#and of course rating

#We have to create test and train set.
#Here contra to Pareto principle we will use a ratio 90/10 train / set in order to 
#have a test set comparable to validation one.

test_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
test_set <- edx%>% slice(test_index)
train_set<- edx%>% slice(-test_index)

#We decided to create new variables and test our model with them. Of course we will apply changes in both sets.
#NRPM (number of ratings per movie), NRPU (number of ratings per user), MA (movie age - difference between year of rating and production year), MR (month of rating), HR (hour of rating).

train_set<-train_set%>% mutate(Rating_date=as.Date(as.POSIXct(timestamp, origin = "1970-01-01") ))
train_set<-train_set%>% mutate(RY=as.numeric(format(Rating_date, '%Y')))
test_set<-test_set%>% mutate(Rating_date=as.Date(as.POSIXct(timestamp, origin = "1970-01-01") ))
test_set<-test_set%>% mutate(RY=as.numeric(format(Rating_date, '%Y')))

#To find movie age (MA) we have to reduce Rating year (RY) by movie year(MY)

train_set<-train_set%>%mutate(MY=substr(title,nchar(title)-4,nchar(title)-1))
train_set<- train_set%>%mutate(MA=RY-as.numeric(MY))
test_set<-test_set%>%mutate(MY=substr(title,nchar(title)-4,nchar(title)-1))
test_set<- test_set%>%mutate(MA=RY-as.numeric(MY))

#To make lighter our dataset at this point we throw away the following columns that we will not use or no need them anymore: title, MY, RY

test_set$title<-NULL
test_set$MY<-NULL
test_set$RY<-NULL
train_set$title<-NULL
train_set$MY<-NULL
train_set$RY<-NULL
test_set<-test_set%>% mutate(MR=as.numeric(format(Rating_date, '%m')))
test_set<-test_set%>% group_by(userId) %>% mutate(NRPU=n())
test_set<-test_set%>% group_by(movieId)%>% mutate(NRPM=n())
train_set<-train_set%>% mutate(MR=as.numeric(format(Rating_date, '%m')))
train_set<-train_set%>% group_by(userId) %>% mutate(NRPU=n())
train_set<-train_set%>% group_by(movieId)%>% mutate(NRPM=n())

#So we have all our variables except the Hour of Rating HR, that we will extract from timestamp
#Lets throw away Rating_date

train_set$Rating_date<-NULL
train_set<-train_set%>% mutate(Rating_date_time=as.POSIXct(timestamp, origin="1970-01-01"))
test_set$Rating_date<-NULL
test_set<-test_set%>% mutate(Rating_date_time=as.POSIXct(timestamp, origin="1970-01-01"))

#So we have extract from timestamp(UNIX mode), Rating_date_time

train_set<-train_set%>% mutate(HR=as.numeric(format(Rating_date_time, '%H')))
test_set<-test_set%>% mutate(HR=as.numeric(format(Rating_date_time, '%H')))

#So now we have all our variables and we can delete timestamp and Rating_date_time
train_set$timestamp<-NULL
train_set$Rating_date_time<-NULL
test_set$timestamp<-NULL
test_set$Rating_date_time<-NULL

#Lets check now if our new variables affect rating

#No need to use the whole edx or train_set 8-9 million lines for this check. Lets use a sample of it.
train_set_sample<-train_set %>% sample_frac(0.1)
dim(train_set_sample)

train_set_sample%>%group_by(NRPU)%>%summarize(NRPU,AVG_RATING=mean(rating))%>%ggplot(aes(NRPU,AVG_RATING))+geom_point()+geom_smooth(method = "lm")

#We see the more a user do rating the stricter is, that sounds normal. From few rating and average more than 3.5 we reach users with 3000 ratings and a bit less than 3 average

train_set_sample%>%group_by(NRPM)%>%summarize(NRPM,AVG_RATING=mean(rating))%>%ggplot(aes(NRPM,AVG_RATING))+geom_point()+geom_smooth(method = "lm")

#Here the more ratings has a movie the higher they are, that also sounds normal as masterpieces take more ratings in average and in number.

train_set_sample%>%group_by(MA)%>%summarize(MA,AVG_RATING=mean(rating))%>%ggplot(aes(MA,AVG_RATING))+geom_point()+geom_smooth(method = "lm")

#The older the movie the better ratings. Expected as older movies are considered of higher quality. So we start from 3.5 rating in new ones and reach 4 to the older ones.

train_set_sample%>%group_by(MR)%>%summarize(MR,AVG_RATING=mean(rating))%>%ggplot(aes(MR,AVG_RATING))+geom_point()+geom_smooth(method = "lm")

#Here we see higher ratings in period Nov-Jan and lower the rest months, but not big differences.

train_set_sample%>%group_by(HR)%>%summarize(HR,AVG_RATING=mean(rating))%>%ggplot(aes(HR,AVG_RATING))+geom_point()+geom_smooth(method = "lm")

#Here we do not see really affectation  by the hour or rating to rating itself. So we will not test this variable in our models.

#Another variable that we will use as predictor is genres. We think type affects rating.
#Lets find out how many unique categories genres we have.

length(unique(edx$genres))

#There are almost 800 categories and we have 9 millions samples. So no reason to break
#down genres, we will use it as it is.

#So we are ready to pass to build our model.
#Our target is RMSE

RMSE <- function(predicted_ratings, true_ratings){
sqrt(mean((true_ratings - predicted_ratings)^2))}

#We start calculating the average rating.

mu <- mean(train_set$rating)

#calculate f_m (factor movie) on the training set

movie_avgs <- train_set %>%group_by(movieId) %>% summarize(f_m = mean(rating - mu))
predicted_ratings_f_m <- mu + test_set %>% left_join(movie_avgs, by='movieId') %>%.$f_m

#Set mu any possible NA that could destroy our RMSE calculation (one is enough).

predicted_ratings_f_m[is.na(predicted_ratings_f_m)]<-mu

RMSE(predicted_ratings_f_m,test_set$rating) 
  
#user
user_avgs <- train_set %>% left_join(movie_avgs, by='movieId') %>%group_by(userId) %>%
summarize(f_u = mean(rating - mu - f_m))

predicted_ratings_f_u <- test_set %>% 
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
mutate(pred = mu + f_m + f_u) %>%
.$pred

predicted_ratings_f_u[is.na(predicted_ratings_f_u)]<-mu
RMSE(predicted_ratings_f_u,test_set$rating)

#genres
genres_avgs <- train_set %>%  
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
group_by(genres) %>%
summarize(f_g = mean(rating - mu - f_m - f_u))

predicted_ratings_f_g <- test_set %>% 
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
left_join(genres_avgs, by='genres') %>%
mutate(pred = mu + f_m + f_u + f_g) %>%
.$pred

predicted_ratings_f_g[is.na(predicted_ratings_f_g)]<-mu
RMSE(predicted_ratings_f_g,test_set$rating)

#add age

ma_avgs <- train_set %>%
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
left_join(genres_avgs, by='genres') %>%
group_by(MA) %>%
summarize(f_ma = mean(rating - mu - f_u - f_m-f_g))

predicted_ratings_f_a <- test_set %>% 
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
left_join(genres_avgs, by='genres') %>%
left_join(ma_avgs,by='MA') %>%
mutate(pred = mu + f_m + f_u + f_g+f_ma) %>%
.$pred


predicted_ratings_f_a[is.na(predicted_ratings_f_a)]<-mu
RMSE(predicted_ratings_f_a,test_set$rating)


#NRPU
nrpu_avgs <- train_set %>%
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
left_join(genres_avgs, by='genres') %>%
left_join(ma_avgs,by='MA') %>%
group_by(NRPU)%>%
summarize(f_nrpu = mean(rating - mu - f_u - f_m-f_g-f_ma))

predicted_ratings_f_nrpu <- test_set %>% 
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
left_join(genres_avgs, by='genres') %>%
left_join(ma_avgs,by='MA') %>%
left_join(nrpu_avgs,by='NRPU')%>%
mutate(pred = mu + f_m + f_u + f_g+f_ma+f_nrpu) %>%
.$pred

predicted_ratings_f_nrpu[is.na(predicted_ratings_f_nrpu)]<-mu

RMSE(predicted_ratings_f_nrpu,test_set$rating)

#Get worse our results. 
#We throw it away and try NRPM


nrpm_avgs <- train_set %>%
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
left_join(genres_avgs, by='genres') %>%
left_join(ma_avgs,by='MA') %>%
group_by(NRPM)%>%
summarize(f_nrpm = mean(rating - mu - f_u - f_m-f_g-f_ma))

predicted_ratings_f_nrpm <- test_set %>% 
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
left_join(genres_avgs, by='genres') %>%
left_join(ma_avgs,by='MA') %>%
left_join(nrpm_avgs,by='NRPM')%>%
mutate(pred = mu + f_m + f_u + f_g+f_ma+f_nrpm) %>%
.$pred

predicted_ratings_f_nrpm[is.na(predicted_ratings_f_nrpm)]<-mu

RMSE(predicted_ratings_f_nrpm,test_set$rating)

#Again the same happens. So we will remain with movieId, UserId, genres and MA (movie age).

#The only that remains is regularization

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  f_m <- train_set %>%
    group_by(movieId) %>%
    summarize(f_m = sum(rating - mu)/(n()+l),.groups='drop')
  f_u <- train_set %>% 
    left_join(f_m, by="movieId") %>%
    group_by(userId) %>%
    summarize(f_u = sum(rating - f_m - mu)/(n()+l),.groups='drop')
  f_g <- train_set %>% 
    left_join(f_m, by="movieId") %>%
    left_join(f_u, by="userId") %>%
    group_by(genres) %>%
    summarize(f_g= sum(rating - f_m - f_u-mu)/(n()+l),.groups='drop')
  f_a<- train_set %>% 
    left_join(f_m, by="movieId") %>%
    left_join(f_u, by="userId") %>%
    left_join(f_g, by="genres") %>%
    group_by(MA) %>%
    summarize(f_a= sum(rating - f_m - f_u - f_g - mu)/(n()+l),.groups='drop')
  predicted_ratings <- 
    test_set %>% 
    left_join(f_m, by = "movieId") %>%
    left_join(f_u, by = "userId") %>%
    left_join(f_g, by = "genres") %>%
    left_join(f_a, by = "MA") %>%
    mutate(pred = mu + f_m + f_u + f_g + f_a) %>%
    .$pred
  predicted_ratings[is.na(predicted_ratings)]<-mu
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses) 

#The lamda that gives min rmse and the min rmse are

lamda<-lambdas[which.min(rmses)]
min(rmses)

#As we have reached our final model it is time prepare validation set.
#We have to bring it to the same form as train or test sets.
#Of course no need do the steps for NRPM or NRPU that finally we do not use.
#The same for HR or MR.

validation<-validation%>% mutate(Rating_date=as.Date(as.POSIXct(timestamp, origin = "1970-01-01") ))
validation<-validation%>% mutate(RY=as.numeric(format(Rating_date, '%Y')))
validation<-validation%>%mutate(MY=substr(title,nchar(title)-4,nchar(title)-1))
validation<- validation%>%mutate(MA=RY-as.numeric(MY))
validation$title<-NULL
validation$MY<-NULL
validation$RY<-NULL
validation$timestamp<-NULL
validation$Rating_date<-NULL

#So lets apply our final model to validation set

mu <- mean(edx$rating)
f_m <- train_set %>%
  group_by(movieId) %>%
  summarize(f_m = sum(rating - mu)/(n()+lamda),.groups='drop')
f_u <- train_set %>% 
  left_join(f_m, by="movieId") %>%
  group_by(userId) %>%
  summarize(f_u = sum(rating - f_m - mu)/(n()+lamda),.groups='drop')
f_g <- train_set %>% 
  left_join(f_m, by="movieId") %>%
  left_join(f_u, by="userId") %>%
  group_by(genres) %>%
  summarize(f_g= sum(rating - f_m - f_u-mu)/(n()+lamda),.groups='drop')
f_a<- train_set %>% 
  left_join(f_m, by="movieId") %>%
  left_join(f_u, by="userId") %>%
  left_join(f_g, by="genres") %>%
  group_by(MA) %>%
  summarize(f_a= sum(rating - f_m - f_u - f_g - mu)/(n()+lamda),.groups='drop')
predicted_ratings <- 
  validation %>% 
  left_join(f_m, by = "movieId") %>%
  left_join(f_u, by = "userId") %>%
  left_join(f_g, by = "genres") %>%
  left_join(f_a, by = "MA") %>%
  mutate(pred = mu + f_m + f_u + f_g + f_a) %>%
  .$pred
predicted_ratings[is.na(predicted_ratings)]<-mu
RMSE(predicted_ratings, validation$rating)

#Analytical comments on the rmd or pdf reports.
