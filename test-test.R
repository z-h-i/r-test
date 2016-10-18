library(rpart) # for decision tree
library(e1071) # for naive bayes' and svm
library(nnet)  # for perceptron/ANN


files <- c('bach_cleaned.csv', 'cancer_cleaned.csv', 'car_cleaned.csv', 'haberman_survival.csv', 'transfusion_cleaned.csv')


## set algorithms' functions
perceptron <- function(formula_x, data_x) {nnet(formula_x, data_x, size = 1, skip = TRUE, maxit = 250, MaxNWts = 2500)}
net <- function(formula_x, data_x) {nnet(formula_x, data_x, size = 14, maxit = 250, MaxNWts = 2500)}
bayes <- function(formula_x, data_x) {naiveBayes(formula_x, data_x, laplace = .0001)}
tree <- function(formula_x, data_x) {rpart(formula_x, data_x, method = 'class')}
svmachine <- function(formula_x, data_x) {svm(formula_x, data_x)}


## and data and functions into a neat list
models <- c(perceptron, net, bayes, tree, svmachine)
formulas <- c(F_M ~ ., Class ~ ., unacc ~ ., X1.1 ~ ., class ~ .)
all_data <- sapply(files, read.csv, simplify = TRUE)


## get rid of index row 'X' and change target column to factor
for (i in 1:5) {all_data[[i]]$X <- NULL}
for (i in 1:5) {all_data[[i]][ ,length(all_data[[i]])] <- as.factor(all_data[[i]][ ,length(all_data[[i]])])}


## grab row counts and ensure row numbers are nice and consistent
rows <- sapply(all_data, nrow)
mapply((function(x, y) {row.names(x) <- 1:y}), x = all_data, y = rows)


## sample (80 training, 20 test) and apply to new training and test sets
samples <- sapply(rows, (function(x) {sample(x, x * .8)}), simplify = TRUE)
train_1 <- mapply((function(x, y) {return(x[y, ])}), x = all_data, y = samples)
test_1 <- mapply((function(x, y) {return(x[-y, ])}), x = all_data, y = samples)


## create a new test set that doesn't overlap with the first test set,
## then generate the corresponding training set
rows_2 <- sapply(train_1, (function(x) {as.numeric(row.names(x))}))
samples_2 <- sapply(rows_2, (function(x) {sample(x, length(x) * .25)}), simplify = TRUE)
test_2 <- mapply((function(x, y) {return(x[y, ])}), x = all_data, y = samples_2)
train_2 <- mapply((function(x, y) {return(x[-y, ])}), x = all_data, y = samples_2)


## 'standardize' the predict function for all algorithms' predict functions,
## then put them into a nice iterable
pred_class <- function(mod_x, test_x) {predict(mod_x, test_x, type = 'class')}
pred_val <- function(mod_x, test_x) {round(predict(mod_x, test_x))}
do_predict <- c(pred_val, pred_val, pred_class, pred_class, pred_class)


## containers to store the results table's info
accuracy_1 <- c()
accuracy_2 <- c()
train_acc_1 <- c()
train_acc_2 <- c()


## make all models, generate the data
for (i in 1:length(models))   # for each model
{
  for (j in 1:length(all_data))   # and for each dataset
  {
    model_1 <- models[[i]](formulas[[j]], train_1[[j]])     # make two models for the two train/test splits
    model_2 <- models[[i]](formulas[[j]], train_2[[j]]) 
    prediction_1 <- do_predict[[i]](model_1, test_1[[j]])   # generate test predictions for each split
    prediction_2 <- do_predict[[i]](model_2, test_2[[j]])
    train_pre_1 <- do_predict[[i]](model_1, train_1[[j]])   # generate training data hypotheses for comparison
    train_pre_2 <- do_predict[[i]](model_2, train_2[[j]])
    accuracy_1 <- c(accuracy_1, sum(prediction_1 == test_1[[j]][ ,length(test_1[[j]])]) / nrow(test_1[[j]]))      # compute simple accuracy measure
    accuracy_2 <- c(accuracy_2, sum(prediction_2 == test_2[[j]][ ,length(test_2[[j]])]) / nrow(test_2[[j]]))
    train_acc_1 <- c(train_acc_1, sum(train_pre_1 == train_1[[j]][ ,length(train_1[[j]])]) / nrow(train_1[[j]]))  # for both training and test predictions
    train_acc_2 <- c(train_acc_2, sum(train_pre_2 == train_2[[j]][ ,length(train_2[[j]])]) / nrow(train_2[[j]]))
  }
}


## prepare columns for the results table
files_2 <- sapply(files, function(x) {paste(x, ' - set 2')})
files <- sapply(files, function(x) {paste(x, ' - set 1')})
lengths <- sapply(all_data, (function(x) {return(length(x) - 1)}))
splits <- c()
for (i in 1:10) {splits <- c(splits, '80/20')}
index <- {x <- 1:25; x[x %% 5 == 1]}

## throw everything into the results table, then generate a csv
results <- data.frame(matrix(NA, nrow = 10, ncol = 0))

results <- cbind(results, c(files, files_2), c(rows, rows), c(lengths, lengths), splits)
for (i in index) {results <- cbind(results, c(accuracy_1[i:(i+4)], accuracy_2[i:(i+4)]))}
for (i in index) {results <- cbind(results, c(train_acc_1[i:(i+4)], train_acc_2[i:(i+4)]))}
colnames(results) <- c('Dataset', 'Instances', 'Attributes', 'Train/Test Split', 'Perceptron Test Accuracy', 
                       'ANN Test Accuracy', 'Naive Bayes Test Accuracy', 'Decision Tree Test Accuracy', 
                       'SVM Test Accuracy', 'Perceptron Train Accuracy', 'ANN Train Accuracy', 
                       'Naive Bayes Train Accuracy', 'Decision Tree Train Accuracy', 
                       'SVM Train Accuracy')
write.csv(results, 'resuts2.csv')
