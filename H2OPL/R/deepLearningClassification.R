#################################################################################
# author: "Alexandre Vilcek"
#################################################################################

library(caret)
library(h2o)


data <- read.csv('H2OPL/data/pml-training.csv', stringsAsFactors=F)
str(data)

na_counts <- apply(data, 2, function(x) length(which(is.na(x))))
plot(na_counts)

es_counts <- apply(data, 2, function(x) length(which(x=='')))
plot(es_counts)

pred_names <- intersect(names(which(na_counts==0)), names(which(es_counts==0)))

pred_names[1:7]

pred_names <- pred_names[8:length(pred_names)]

data <- data[pred_names]
names(data)

set.seed(102030)
train <- list()
test <- list()
for(i in 1:3) {
  index_train <- createDataPartition(y=data$classe, p=0.6, list=F) 
  train[[i]] <- data[index_train,]
  test[[i]] <- data[-index_train,]
}

h2o_cluster <- h2o.init(ip = 'localhost', port =54321, max_mem_size = '6g', nthreads=8)

### Model 1: 2-Layer Deep Neural Network ------------------------------------------------------------------

deeplearning_models <- list()
for(i in 1:3) {

  ##########################################################################################################
  ##-----------------------------------Why as.h2o doesn't work ???----------------------------------------##
  # h2o_train_path <- "/Users/uncxu/DeepLearningByLearningCodeFromOthers/H2OPL/data/pml-training.csv"
  # h2o_test_path <- "/Users/uncxu/DeepLearningByLearningCodeFromOthers/H2OPL/data/pml-testing.csv"
  # h2o_train <- h2o.uploadFile(h2o_cluster, path = h2o_train_path)
  # h2o_test <- h2o.uploadFile(h2o_cluster, path = h2o_test_path)
  # h2o_train <- as.h2o(train[[i]], h2o_cluster, "h2o_train")
  # h2o_test <- as.h2o(test[[i]], h2o_cluster, "data_test")
  ##########################################################################################################
  
  write.csv(train[[i]], file = "H2OPL/data/train.csv",row.names=FALSE)
  h2o_train_path <- "/Users/uncxu/DeepLearningByLearningCodeFromOthers/H2OPL/data/train.csv"
  write.csv(test[[i]], file = "H2OPL/data/test.csv",row.names=FALSE)
  h2o_test_path <- "/Users/uncxu/DeepLearningByLearningCodeFromOthers/H2OPL/data/test.csv"
  
  h2o_train <- h2o.uploadFile(h2o_cluster, path = h2o_train_path)
  h2o_test <- h2o.uploadFile(h2o_cluster, path = h2o_test_path)
  
  deeplearning_model <- h2o.deeplearning(x=names(data[1:52]),
                                         y=names(data[53]),
                                         training_frame=h2o_train,
                                         validation_frame=h2o_test,
                                         overwrite_with_best_model=T,
                                         activation='RectifierWithDropout',
                                         hidden=c(100,100),
                                         adaptive_rate=T,
                                         rho=0.99,
                                         epsilon=1e-8,
                                      #   nesterov_accelerated_gradient=T,
                                      #   input_dropout_ratio=0.2,
                                      #   hidden_dropout_ratios=c(0.5,0.5),
                                         balance_classes=T,
                                         epochs=500)
  deeplearning_models[[i]] <- deeplearning_model
}


### Model 2: Random Forest -----------------------------------------------------------

randomforest_models <- list()
#for(i in 1:3) {
#  h2o_data_train <- as.h2o(h2o_cluster, train[[i]], 'data_train')
#  h2o_data_test <- as.h2o(h2o_cluster, test[[i]], 'data_test')
  randomforest_model <- h2o.randomForest(x=names(data[1:52]),
                                         y=names(data[53]),
                                         training_frame=h2o_train,
                                         validation_frame=h2o_test,
                                         ntree=100,
                                         depth=50,
                                         balance.classes=T,
                                         type='BigData')
  randomforest_models[[i]] <- randomforest_model
#}

### Model 3 Gradient Boosted Machine -------------------------------------------------------

gradientboosted_models <- list()
#for(i in 1:3) {
#  h2o_data_train <- as.h2o(h2o_cluster, train[[i]], 'data_train')
#  h2o_data_test <- as.h2o(h2o_cluster, test[[i]], 'data_test')
  gradientboosted_model <- h2o.gbm(x=names(data[1:52]),
                                   y=names(data[53]),
                                   training_frame=h2o_train,
                                   validation_frame=h2o_test,
                                   n.trees=100,
                                   interaction.depth=50,
                                   balance.classes=T)
  gradientboosted_models[[i]] <- gradientboosted_model
#}


### Model training results ----------------------------------------------------------------

model_names <- c('Deep Neural Network', 'Random Forest', 'Gradient Boosted Model')
acc_dnn <- vector()
acc_rf <- vector()
acc_gbm <- vector()
#for(i in 1:3) {
acc_dnn <- cbind(acc_dnn, deeplearning_models[[i]]@model$validation_metrics@metrics$MSE)
acc_rf <- cbind(acc_rf, randomforest_models[[i]]@model$validation_metrics@metrics$MSE)
  acc_gbm <- cbind(acc_gbm, gradientboosted_models[[i]]@model$validation_metrics@metrics$MSE)
#}
models <- data.frame(model_names, c(mean(acc_dnn), mean(acc_rf), mean(acc_gbm)))
names(models) <- c('Model.Name', 'Model.Accuracy')
models

acc_rf

### Model predicting results ----------------------------------------------------------------

test <- read.csv('H2OPL/data/pml-testing.csv', stringsAsFactors=F)
test <- test[pred_names[1:52]]

#h2o_data_test <- as.h2o(h2o_cluster, test, 'data_test')
write.csv(test, file = "H2OPL/data/test.csv",row.names=FALSE)
h2o_test_path <- "/Users/uncxu/DeepLearningByLearningCodeFromOthers/H2OPL/data/test.csv"
h2o_test <- h2o.uploadFile(h2o_cluster, path = h2o_test_path)
h2o_model <- deeplearning_models[[3]]
predictions <- h2o.predict(h2o_model, h2o_test)
predictions <- as.data.frame(predictions)$predict

# generate files for submission
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(predictions)


#############################################################
# Another example
# 
#############################################################

h2o_server = h2o.init()

## It seems uploadFile and importFile both upload the file to h2o cloud
iris_train <- read.table("H2OPL/data/iris.txt", sep=",")
iris_test <- iris_train[,-5]
write.csv(iris_train, file = "H2OPL/data/iris_train.csv", row.names = FALSE)
write.csv(iris_test, file = "H2OPL/data/iris_test.csv", row.names = FALSE)

iris_train <- h2o.uploadFile(h2o_server, path="/Users/uncxu/DeepLearningByLearningCodeFromOthers/H2OPL/data/iris_train.csv",
                             sep =',',header = T)
iris_test <- h2o.uploadFile(h2o_server, path="/Users/uncxu/DeepLearningByLearningCodeFromOthers/H2OPL/data/iris_test.csv",
                             sep =',',header = T)

iris_model <- h2o.deeplearning( x = 1:4, y = 5, training_frame = iris_train, activation = "Tanh", hidden = c(50,50,50), epochs =500)

# prediction
iris_pred <- h2o.predict(iris_model, iris_test)
iris_predict <- as.data.frame(iris_pred)$predict

# compare the prediction with the true
iris_train <- as.data.frame(iris_train)
# prediction error
sum(iris_train[,5] != iris_predict)/length(iris_predict)
# [1] 0.06



