# Process Single File

data <- read.csv("Expr_ClinData_marker.txt", sep="\t")
data <- data[!is.na(data[,960]),]
data[,961] <- abs(data[,961]-1)
# cancer types with more than 300 samples
types_filter <- names(table(data[,959])[table(data[,959]) > 300])
data <- data[data[,959] %in% types_filter,]


TrainingData=sample(1:nrow(data),replace = F,size = 0.8*nrow(data))
tmp=setdiff(1:nrow(data),TrainingData)
TestData=sample(tmp,replace = F,size =0.1*nrow(data))
EvalData=setdiff(tmp,TestData)
TrainingData=data[TrainingData,]
TestData=data[TestData,]
EvalData=data[EvalData,]

write.table(EvalData,file = 'EvalData.txt',sep = '\t', row.names = F, quote = F)
write.table(TestData,file = 'TestData.txt',sep = '\t',row.names = F, quote = F)
write.table(TrainingData,file = 'TrainingData.txt',sep = '\t',row.names = F, quote = F)
