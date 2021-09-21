#load the functions
#source("functions.R")
#library(pROC)
library(neuralnet)
require(nnet)

# THREE SIDES

aux<-trainingData[,-ncol(trainingData)]

aux<-cbind(aux,class.ind(trainingData$clase))
colnames(aux)[c(
  ncol(aux)-2,
  ncol(aux)-1,
  ncol(aux)
)]<-c("LEFT","STR","RIGHT")

trainingData<-aux

n <- names(trainingData)
f <- as.formula(paste(
  "LEFT + STR + RIGHT ~",
  paste(n[!n %in% c("LEFT","STR","RIGHT")],
        collapse = " + ")
))

nn_test<-neuralnet(f, trainingData,hidden=c(5,3),linear.output=FALSE, act.fct="logistic")
plot(nn_test)
pred<-compute(nn_test,trainingData[,1:1024])

predicciones<-unlist(pred$net.result)
predicciones<-data.frame(
  predicciones,
  cbind(
    trainingData$LEFT,
    trainingData$STR,
    trainingData$RIGHT
  )
)

for(i in 1:nrow(predicciones)){
  if(predicciones$X1[i]>predicciones$X2[i] && predicciones$X1[i] > predicciones$X3[i])
    predicciones[i,c(1,2,3)]<-c(1,0,0)
  
  if(predicciones$X2[i]>predicciones$X1[i] && predicciones$X2[i] > predicciones$X3[i])
    predicciones[i,c(1,2,3)]<-c(0,1,0)
  
  if(predicciones$X3[i]>predicciones$X2[i] && predicciones$X3[i] > predicciones$X1[i])
    predicciones[i,c(1,2,3)]<-c(0,0,1)
}

acc = sum(
  (predicciones[,1] == predicciones[,4]) *1,
  (predicciones[,2] == predicciones[,5]) *1,
  (predicciones[,3] == predicciones[,6]) *1
)/(nrow(predicciones)*3)

anterior = nn_test

subset_data <- nn_test$result.matrix

out_weigths<-nn_test$result.matrix[4:nrow(nn_test$result.matrix)]

write.csv(out_weigths,file = "weights_three_sides.csv", row.names = FALSE)
