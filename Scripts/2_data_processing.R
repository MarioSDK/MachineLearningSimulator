library(readr)

# READ FILES

prueba <- read_delim(file="images_three_sides/imrows_left.csv", delim=",", col_names = FALSE)
datos_prueba1<-data.frame(cbind(prueba[,],rep(0,nrow(prueba))))

prueba <- read_delim(file="images_three_sides/imrows_straight.csv", delim=",", col_names = FALSE)
datos_prueba2<-data.frame(cbind(prueba[,],rep(1,nrow(prueba))))

prueba <- read_delim(file="images_three_sides/imrows_right.csv",delim=",", col_names = FALSE)
datos_prueba3<-data.frame(cbind(prueba[,],rep(2,nrow(prueba))))

# COLNAMES

colnames(datos_prueba1)<-c(paste0("X",seq(1,ncol(datos_prueba1)-1, by=1)),"clase")
colnames(datos_prueba2)<-c(paste0("X",seq(1,ncol(datos_prueba1)-1, by=1)),"clase")
colnames(datos_prueba3)<-c(paste0("X",seq(1,ncol(datos_prueba1)-1, by=1)),"clase")

# TRAINING DATA

trainingData <- rbind(
  datos_prueba1,
  datos_prueba2,
  datos_prueba3
)
