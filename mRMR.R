#code for running mRMRe
install.packages("mRMRe")
library(mRMRe)

#data<-read.csv('C:/Users/training.csv',header=T)
data = feature_lable[,-1]
feature_num = ncol(data) - 1
train_feature = data[,0:feature_num] 
train_label = data[,585]
mrmr_feature<-train_feature
mrmr_feature$y<-train_label
target_indices = which(names(mrmr_feature)=='y')
for (m in which(sapply(mrmr_feature, class)!="numeric"))
{mrmr_feature[,m]=as.numeric(unlist(mrmr_feature[,m]))
}
Data <- mRMR.data(data = data.frame(mrmr_feature))
mrmr=mRMR.ensemble(data = Data, target_indices = target_indices, 
                   feature_count = 150, solution_count = 1)
index=mrmr@filters[[as.character(mrmr@target_indices)]]
new_data <- nrow(data):ncol(index)
dim(new_data) <- c(nrow(data),ncol(index))
data = as.data.frame(data)
new_data = data[,index]
new_data_0 = cbind(new_data,train_label)
write.csv (new_data_0,"features after mRMR.csv",row.names = F)
