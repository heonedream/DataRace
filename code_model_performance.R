setwd("/Users/hewenjun/Desktop/银联银杏大数据竞赛")
#读取数据
model_sample <- read.csv(file = "model_sample.csv")
str(model_sample)


#查看一下缺失情况
library(VIM)
aggr(model_sample,prop=FALSE,numbers=TRUE)


data.missing <- is.na(model_sample)
#缺失个数多于10000的变量
a <- c()
for (i in 1:length(names(model_sample))) {
  Missing <- sum(data.missing[,i])
  if(Missing>=10000) {a <- c(a,i);
  print(names(model_sample)[i]);
  print(Missing)
  }
}
names(model_sample)[a]


#查看是否是一个不平衡数据
table(model_sample$y)#是一个不平衡数据


#处理缺失
#拟对于分类变量而言，缺失的变量划分为一类；对于连续型的变量根据情况而定
#可以把缺失当做一类的变量，并设置为因子，其他均为连续型变量
#以下是根据某些规则评价出来存在缺失的变量，没有注销的可以看做是分类变量
c("x_042","x_049","x_056","x_062","x_063",
  "x_065","x_066","x_068","x_069","x_071","x_072","x_075","x_081","x_082",
  #"x_088",
  "x_089","x_092","x_093","x_094","x_096","x_097","x_100","x_101",
  "x_102","x_103","x_104","x_105","x_106","x_107","x_109","x_110","x_112",
  "x_113","x_115","x_116","x_118","x_119","x_120","x_122","x_128","x_129",
  "x_132","x_134","x_137","x_139","x_142",
  #"x_144",
  "x_149","x_150","x_151",
  "x_152","x_154",
  #"x_155","x_156","x_157","x_158",
  "x_162","x_163","x_164",
  "x_165",
  #"x_166",
  "x_175","x_176","x_177","x_178","x_188","x_189","x_190",
  "x_191","x_192","x_193"
  #,"x_194","x_195","x_196","x_197","x_198","x_199"
)
#对分类变量进行赋值后，划分为因子的一类,有探索性分析可得
for (i in c("x_042","x_049","x_056","x_062","x_063",
            "x_065","x_066","x_068","x_069","x_071","x_072","x_075","x_081","x_082",
            #"x_088",
            "x_089","x_092","x_093","x_094","x_096","x_097","x_100","x_101",
            "x_102","x_103","x_104","x_105","x_106","x_107","x_109","x_110","x_112",
            "x_113","x_115","x_116","x_118","x_119","x_120","x_122","x_128","x_129",
            "x_132","x_134","x_137","x_139","x_142",
            #"x_144",
            "x_149","x_150","x_151",
            "x_152","x_154",
            #"x_155","x_156","x_157","x_158",
            "x_162","x_163","x_164",
            "x_165",
            #"x_166",
            "x_175","x_176","x_177","x_178","x_188","x_189","x_190",
            "x_191","x_192","x_193"
            #,"x_194","x_195","x_196","x_197","x_198","x_199"
)) {
  model_sample[,i][is.na(model_sample[,i])] <- 0.5
  model_sample[,i] <- factor(model_sample[,i])
}

#把变量x_001到x_041的分类变量转成因子
#找出这些变量的positions
factor_name <- c()
for (i in 3:42){
  if(!(names(model_sample)[i] %in% c("x_002","x_020","x_021","x_025","x_026","x_029","x_030","x_031",
                                     "x_034","x_035"))){
    factor_name <- c(factor_name,names(model_sample)[i])
  }
}
#找出应为为因子的变量
factor_name <- c("x_001","x_003","x_004","x_005","x_006","x_007","x_008","x_009","x_010","x_011","x_012","x_013","x_014",
                 "x_015","x_016","x_017","x_018","x_019","x_022","x_023","x_024","x_027","x_028","x_032","x_033","x_036",
                 "x_037","x_038","x_039","x_040")
#设置为factor
for (i in factor_name) {
  model_sample[,i] <- factor(model_sample[,i])
  
}
#把缺失数达到8000的变量去
data.missing <- is.na(model_sample)
a <- c()
for (i in 1:length(names(model_sample))) {
  Missing <- sum(data.missing[,i])
  if(Missing>=8000) {a <- c(a,i);
  print(names(model_sample)[i]);
  print(Missing)
  }
}
model_sample <- model_sample[,-a]
##剩下的连续型变量的缺失值只能用于插补了
##进行bagged trees缺失值填补，但是出现报错
data.model.pre <- model_sample[,-c(1,2)]#应该把y也提到，因为官方给的测试集也没有y,但缺失值填补过程需要集合测试集来做
library(caret)
#preProcValues <- preProcess(data.model.pre, method = c("bagImpute"))#存在报错
preProcValues <- preProcess(data.model.pre, method = c("knnImpute"))#没有报错
data.model <- predict(preProcValues, data.model.pre)
str(data.model)



##看看需不需要把zero or Zero-variance predictors去掉
nzv <- nearZeroVar(data.model)
data.model <- data.model[,-nzv]
data.model$y <- model_sample$y



##这里就应该分出测试集合训练集
#从原始数据中抽一个测试集出来
#要从更改factor names之后再分开
set.seed(123)
data.train.number <- createDataPartition(data.model$y,p=0.8,list = F,times = 1)
data.train <- data.model[data.train.number,]
data.test <- data.model[-data.train.number,]



##进行单变量筛选，进一步剔除噪音变量
p=c();item=c()
for(i in 1:(length(names(data.train))-1)){
  model.logi <- glm(y ~data.train[,i], data=data.train, family=binomial(link="logit"))
  p.model.logi <- summary(model.logi)
  item=c(item,i)
  p <- c(p,p.model.logi$coefficients[2,4])
}
univarite.result <- data.frame(cbind(item,p))
#write.csv(univarite.result,"univariate.csv")
keep <- univarite.result[univarite.result$p<0.05,"item"]
data.train <- data.train[,c(keep,length(names(data.train)))]
data.train$y <- as.factor(data.train$y)


##看看是否需要找相关的Predictor，针对连续性变量
x_numeric<-sapply(data.train,is.numeric)
myda<-data.train[,x_numeric,drop=FALSE]
descrCor <- cor(myda)
highlyCorDescr <- findCorrelation(descrCor,cutoff = 0.75)
#去掉相关系数大于0.75的变量
d <- c()
for (i in 1:length(names(data.train))) {
  if(!(names(data.train)[i] %in% names(myda)[highlyCorDescr])) d <- c(d,i)
  
}
data.train <- data.train[,d]



#线性组合的Linear dependencies,暂时不考虑
#缺失值填补后，仍有些值是缺失的
data.train <- na.omit(data.train)


#更改factor名字，以免后面进行caret训练的时候出错
feature.names=names(data.train)
for (f in feature.names) {
  if (class(data.train[[f]])=="factor") {
    levels <- unique(c(data.train[[f]]))
    data.train[[f]] <- factor(data.train[[f]],
                              labels=make.names(levels))
  }
}


############################################train
##探索采用prSummary,可以使用
##由于结果F1的缘故，我们把data.train的解决变量换一下X1代表1，X2代表2
data.train$y <- as.character(data.train$y)
data.train$y <- factor(data.train$y,levels = c("X2","X1"))



##测试结果显示是sampling 的方式是"up"比较好,但是有可能真实的状况是“down”更加好
#模型训练因为是不平衡，所有需要悬着up sampling，down sampling还有另外两种。
#当模型不那么复杂的时候，用一下的fitControl是挺好的
# fitControl <- trainControl(
#   method = "repeatedcv",
#   number = 10,
#   repeats = 10,
#   classProbs = T,
#   allowParallel =T,
#   summaryFunction = f1,
#   sampling = "down"
# )
#为了减少模型训练时间，用单纯的cv，选择prSummary(Recall,precision,F1)
fitControl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = T,
  allowParallel =T,
  summaryFunction = prSummary,
  sampling = "up",
  search = "random"
)
set.seed(128)
#然后metric选定F1,每个参数选5个
xgb.upside.final <- train(y~.,data = data.model,
                          method='xgbTree',
                          trControl=fitControl,
                          tuneLength=5,
                          verbose=F,
                          metric="F")
xgb.upside.final



#选定最终的参数后进行如下模型
fitControl.final <- trainControl(method = 'none',classProbs = T)
set.seed(128)
xgb.model <- train(y~.,data = data.train,
                   method='xgbTree',
                   trControl=fitControl.final,
                   verbose=F,
                   tuneGrid=data.frame(nrounds = 849,
                                       max_depth =8,
                                       eta = 0.03130453,
                                       gamma=6.265423,
                                       colsample_bytree =0.3185898,
                                       min_child_weight = 3,
                                       subsample = 0.342483),
                   metric="F")

#对预测集进行估计
data.class <- predict(xgb.model,newdata=data.test)
table(data.class,data.test$y)
