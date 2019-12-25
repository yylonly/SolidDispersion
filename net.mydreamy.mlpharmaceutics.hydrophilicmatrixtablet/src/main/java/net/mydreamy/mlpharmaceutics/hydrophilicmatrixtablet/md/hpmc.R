library(mlbench)
library(caret)

rm(list = ls())

allX<- read.csv("~/Desktop/PharmaceuticsData/HPMC/alldata.csv")

##range 0 to 1
alldata <- data.matrix(allX)
y <- alldata[,19:22]/100
X <- alldata[, 1:18]

maxs <- apply(X, 2, max)
mins <- apply(X, 2, min)
ranges <- maxs - mins
means <- apply(X, 2, mean)
scaledallx <- scale(X, center = mins, scale = ranges)
scaleddata <- cbind(scaledallx, y)

set.seed(15)

numbers = dim(scaledallx)[1];

## A random sample of 5 data points
initalIndexes <- sample(numbers, 5)
#initalIndexes <- c(5,50,78,99,117)
#initalIndexes <- c(18,64,65,66,84)
#initalIndexes <- c(18,64,65,66,74,83,84)

TrainningSet <- scaledallx[-initalIndexes, ]
initalTestSet <- scaledallx[initalIndexes, ]

SelectedIndex <- maxDissim(initalTestSet, TrainningSet, n = 15)
FinalSelectedSet <- TrainningSet[SelectedIndex, ]

cat("Selected Indexes are: ", SelectedIndex, "\n", sep=",")
write.csv(scaleddata[SelectedIndex,], "~/Desktop/PharmaceuticsData/HPMC/testingset.csv", row.names = FALSE)
write.csv(scaleddata[-SelectedIndex,], "~/Desktop/PharmaceuticsData/HPMC/trainingset.csv", row.names = FALSE)


