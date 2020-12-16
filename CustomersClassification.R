#1. IMPORTAR BIBLIOTECAS NECESSÁRIAS PARA AS ANÁLISES
#caso de algum erro, desinstalar e instalar os pacotes abaixo deve resolver
# tidyverse se for erro relacionado à leitura/edição de bases (%>%)
# caret se for erro relacionado à função "traincontrol"
library(tidyverse)
library(caret)
library(pROC)
library(readxl)
library(MLmetrics)
library(e1071)
library(ranger)
library(DMwR)
library(C50)
library(libcoin)
library(MASS)
library(NB)
library(naivebayes)
library(klaR)
library(broom)


#2. LER O CONJUNTO DE DADOS DIRETAMENTE DA FONTE INDICADA - NESSE CASO DIRETO DO GITHUB
data <- read_csv("https://github.com/ifood/ifood-data-advanced-analytics-test/raw/master/ml_project1_data.csv", locale(encoding = "ISO-8859-1"),col_names = TRUE,col_types = NULL )

#3. INDICAR O CAMINHO ATIVO PARA SALVAR OS ARQUIVOS GERADOS DURANTE A EXECUÇÃO DO cÓDIGO
path <- "C:\\Users\\janai\\Documents\\Food"
setwd(path)

#4. CRIAR FEATURES DE INTERESSE A PARTIR DAS FEATURES EXISTENTES
#a) Criação da Feature Age (em anos) a parir da Feature Year_Birth
data$Age <- year(today()) - data$Year_Birth

#b) Criação da Feature TimeEnrollment (em dias) a partir da Feature Dt_Customer
data$TimeEnrollment <- today() - as_date(data$Dt_Customer)

#c)Criação da Feature SomeCmp a partir das Features AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5 
# Essa Feature é 1 se o Cliente aceitou a oferta de alguma campanha antes da última

#d)Criação da Feature QtdeCmp a partir das Features AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5 
# Essa Feature soma quantas campanhas o clinete aceitou antes da última

data$SomeCmp <- NA
data$QtdeCmp <- NA

for (n in 1:nrow(data)) {
  if (data$AcceptedCmp1[n] ==1 || data$AcceptedCmp2[n] ==1 || data$AcceptedCmp3[n] == 1 || data$AcceptedCmp4[n] == 1 || data$AcceptedCmp5[n] == 1){
    data$SomeCmp[n] <- 1
  } else{
    data$SomeCmp[n] <- 0
  }   
  
  
  data$QtdeCmp[n] <- data$AcceptedCmp1[n] + data$AcceptedCmp2[n] + data$AcceptedCmp3[n] + data$AcceptedCmp4[n] + data$AcceptedCmp5[n]
  
}

#e)Criar a Feautre Count apenas para servir de contador em algumas análises
data$Count <- 1


#5. ALTERAR RÓTULOS DAS CATEGORIAS, TIPOS DE DADOS CONFORME DESEJADO
#a) Fazer cópia dos dados 
data2 <- data

#b)Verificar o Summary para analisar tipos de dados
summary(data2)

#c)Transformar os Dados para o Tipo Desejado
data2$Education <- as.factor(data2$Education)
data2$Marital_Status <- as.factor(data2$Marital_Status)
data2$Kidhome <- as.factor(data2$Kidhome)
data2$Teenhome <- as.factor(data2$Teenhome)
data2$AcceptedCmp1 <- as.factor(data2$AcceptedCmp1)
data2$AcceptedCmp2 <- as.factor(data2$AcceptedCmp2)
data2$AcceptedCmp3 <- as.factor(data2$AcceptedCmp3)
data2$AcceptedCmp4 <- as.factor(data2$AcceptedCmp4)
data2$AcceptedCmp5 <- as.factor(data2$AcceptedCmp5)
data2$Complain <- as.factor(data2$Complain)
data2$Response <- as.factor(data2$Response)
data2$TimeEnrollment <- as.numeric(data2$TimeEnrollment)
data2$SomeCmp <- as.factor(data2$SomeCmp)

#d)Verificar o Summary novamente para verificar se transformação foi bem sucedida
summary(data2)

#e)Remover outliers
remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 3 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

data2$Age <- remove_outliers(data2$Age)
data2$Income <- remove_outliers(data2$Income)


#f) Modificar os dados para que representem os textos desejados
#Assumi a premissa que os Status Alone, YOLO e Absurd podem ser agregados numa categoria Other uma vez que apresentam poucos registros e não sçao uma categoria comum
colunas = c("AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "SomeCmp",
            "Complain")
colunas2 = c("Marital_Status")
colunas3 = c("Response")

data2 = data2 %>%
  mutate_at(vars(colunas), function(x){ case_when(x == "1" ~ "Sim", 
                                                  x == "0" ~ "Nao")})  %>%
  mutate_at(vars(colunas2), function(x){ case_when(x == "Alone" ~ "Other",
                                                   x == "Absurd" ~ "Other",
                                                   x == "YOLO" ~ "Other",
                                                   x == "Divorced" ~ "Divorced",
                                                   x == "Married" ~ "Married",
                                                   x == "Single" ~ "Single",
                                                   x == "Together" ~ "Together",
                                                   x == "Widow" ~ "Widow")}) %>%
  mutate_at(vars(colunas3), function(x){ case_when(x == "1" ~ "Accepted", 
                                                   x == "0" ~ "Not.Accepted")}) %>%
  mutate(Response = factor(Response, levels = c("Accepted", "Not.Accepted"))) 

#g)Retirar registros com dados faltantes. Como nessa base foram poucos casos resolvi remover, mas caso contrário seria necessário fazer um imputação.
data2 <- na.omit(data2)


#6. CRIAR MODELO PREDITIVO DE CLASSIFICAÇÃO
#a) Criar o conjunto de treino e teste
#Os conjuntos serão formados usando difentes técnicas de balanceamento uma vez que a variável de interesse é 
#desbalanceada, ou seja, apenas 15% dos registros Aceitaram a oferta na última campanha
#Técnicas de balancemaento usadas:
#Downsampling e upSampling

#Primeiro separa no conjunto de treino e teste usando o dados originais
set.seed(2 ^ 31 - 1)
trainIndex = createDataPartition(data2$Response,
                                  p = 0.8,
                                  list = FALSE,
                                  times = 1)[,1]

dados_treino = data2[trainIndex, ]
dados_teste = data2[-trainIndex, ]

#Criar diferentes versões do conjunto de treino usando técnicas de balanceamento
##DownSampling
down_train <- downSample(x = dados_treino[, -c(29)],
                         y = dados_treino$Response)
table(down_train$Class)   

##UpSampling
up_train <- upSample(x = dados_treino[, -c(29)],
                     y = dados_treino$Response)                        
table(up_train$Class) 

#b) Definir como será feito o treino do modelo com 10-fold cross validation
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  classProbs = TRUE,
  repeats = 5,
  savePredictions = TRUE,
  summaryFunction = twoClassSummary
)

#c) Treinar o modelo usando cada conjunto de teste criado
#Os modelos rodados serão:
## "LR" -> simple additive logistic regression.
## "NB" -> Naive Bayes

#obs: a mesma estrutura pode ser usada para outros tipo de classificadores como Decision Tree e Random Forest

#### "LR" -> Logistic Regression 
ori_lreg<-train(Response~Age+ Education+ Marital_Status+ Income + TimeEnrollment+ Kidhome+ Teenhome +Recency+ MntWines+ MntFruits+ MntMeatProducts+ MntFishProducts+
              MntSweetProducts+ MntGoldProds+ NumDealsPurchases+ NumWebPurchases+ NumCatalogPurchases+ NumStorePurchases+ NumWebVisitsMonth+ AcceptedCmp1+
              AcceptedCmp2+ AcceptedCmp3+ AcceptedCmp4+ AcceptedCmp5+ SomeCmp +QtdeCmp+ Complain,
            data=dados_treino,
            method="glm",
            family = binomial(),
            metric = "ROC",
            trControl = fitControl)

down_lreg<-train(Class~Age+ Education+ Marital_Status+ Income + TimeEnrollment+ Kidhome+ Teenhome +Recency+ MntWines+ MntFruits+ MntMeatProducts+ MntFishProducts+
                  MntSweetProducts+ MntGoldProds+ NumDealsPurchases+ NumWebPurchases+ NumCatalogPurchases+ NumStorePurchases+ NumWebVisitsMonth+ AcceptedCmp1+
                  AcceptedCmp2+ AcceptedCmp3+ AcceptedCmp4+ AcceptedCmp5+ SomeCmp +QtdeCmp+ Complain,
                data=down_train,
                method="glm",
                family = binomial(),
                metric = "ROC",
                trControl = fitControl)

up_lreg<-train(Class~Age+ Education+ Marital_Status+ Income + TimeEnrollment+ Kidhome+ Teenhome +Recency+ MntWines+ MntFruits+ MntMeatProducts+ MntFishProducts+
                  MntSweetProducts+ MntGoldProds+ NumDealsPurchases+ NumWebPurchases+ NumCatalogPurchases+ NumStorePurchases+ NumWebVisitsMonth+ AcceptedCmp1+
                  AcceptedCmp2+ AcceptedCmp3+ AcceptedCmp4+ AcceptedCmp5+ SomeCmp +QtdeCmp+ Complain,
                data=up_train,
               method="glm",
               family = binomial(),
                metric = "ROC",
                trControl = fitControl)

## "NB" -> Naive Bayes
ori_nb<-train(Response~Age+ Education+ Marital_Status+ Income + TimeEnrollment+ Kidhome+ Teenhome +Recency+ MntWines+ MntFruits+ MntMeatProducts+ MntFishProducts+
                  MntSweetProducts+ MntGoldProds+ NumDealsPurchases+ NumWebPurchases+ NumCatalogPurchases+ NumStorePurchases+ NumWebVisitsMonth+ AcceptedCmp1+
                  AcceptedCmp2+ AcceptedCmp3+ AcceptedCmp4+ AcceptedCmp5+ SomeCmp +QtdeCmp+ Complain,
                data=dados_treino,
                method="nb",
                metric = "ROC",
                trControl = fitControl)

down_nb<-train(Class~Age+ Education+ Marital_Status+ Income + TimeEnrollment+ Kidhome+ Teenhome +Recency+ MntWines+ MntFruits+ MntMeatProducts+ MntFishProducts+
                   MntSweetProducts+ MntGoldProds+ NumDealsPurchases+ NumWebPurchases+ NumCatalogPurchases+ NumStorePurchases+ NumWebVisitsMonth+ AcceptedCmp1+
                   AcceptedCmp2+ AcceptedCmp3+ AcceptedCmp4+ AcceptedCmp5+ SomeCmp +QtdeCmp+ Complain,
                 data=down_train,
                 method="nb",
                 metric = "ROC",
                 trControl = fitControl)

up_nb<-train(Class~Age+ Education+ Marital_Status+ Income + TimeEnrollment+ Kidhome+ Teenhome +Recency+ MntWines+ MntFruits+ MntMeatProducts+ MntFishProducts+
                 MntSweetProducts+ MntGoldProds+ NumDealsPurchases+ NumWebPurchases+ NumCatalogPurchases+ NumStorePurchases+ NumWebVisitsMonth+ AcceptedCmp1+
                 AcceptedCmp2+ AcceptedCmp3+ AcceptedCmp4+ AcceptedCmp5+ SomeCmp +QtdeCmp+ Complain,
               data=up_train,
               method="nb",
               metric = "ROC",
               trControl = fitControl)



##d) Compilar os resultados a partir da predição feita usando o conjunto de teste
models <- list(ori_lreg, down_lreg, up_lreg, ori_nb, down_nb, up_nb)
amostragem <- list("null", "downsampling", "upsampling", "null", "downsampling", "upsampling")

previsao <- predict(models[[1]], dados_teste, type = "prob")[,"Accepted"]
act_num = ifelse(dados_teste$Response == "Accepted", 1, 0)
pred_num<-factor(ifelse(previsao > 0.5, 1, 0))
auc = as.numeric(auc(roc(act_num, previsao)))
testRESULTS<-data.frame(RESULTS = factor(act_num))
matriz_confusao = confusionMatrix(data=pred_num, reference=testRESULTS$RESULTS, positive = levels(testRESULTS$RESULTS)[2])
metricas = data.frame(metodo = models[[1]]$method,
                      Amostragem = amostragem[[1]],
                      Verdadeiro_Negativo = matriz_confusao$table[1],
                      Falso_Positivo = matriz_confusao$table[2],
                      Falso_Negativo = matriz_confusao$table[3],
                      Verdadeiro_Positivo = matriz_confusao$table[4],
                      Acurácia = matriz_confusao$overall[[1]],
                      Kappa = matriz_confusao$overall[[2]],
                      Sensibilidade = matriz_confusao$byClass[[1]],
                      Especificidade = matriz_confusao$byClass[[2]],
                      PPV = matriz_confusao$byClass[[3]],
                      NPV = matriz_confusao$byClass[[4]],
                      Precisão = matriz_confusao$byClass[[5]],
                      Recall = matriz_confusao$byClass[[6]],
                      F1 = matriz_confusao$byClass[[7]],
                      AUC = auc
                      )
                      

for (i in 2:length(models)){
  previsao <- predict(models[[i]], dados_teste, type = "prob")[,"Accepted"]
  act_num = ifelse(dados_teste$Response == "Accepted", 1, 0)
  pred_num<-factor(ifelse(previsao > 0.5, 1, 0))
  auc = as.numeric(auc(roc(act_num, previsao)))
  testRESULTS<-data.frame(RESULTS = factor(act_num))
  matriz_confusao = confusionMatrix(data=pred_num, reference=testRESULTS$RESULTS, positive = levels(testRESULTS$RESULTS)[2])
  metricas_aux = data.frame(metodo = models[[i]]$method,
                            Amostragem = amostragem[[i]],
                            Verdadeiro_Negativo = matriz_confusao$table[1],
                            Falso_Positivo = matriz_confusao$table[2],
                            Falso_Negativo = matriz_confusao$table[3],
                            Verdadeiro_Positivo = matriz_confusao$table[4],
                            Acurácia = matriz_confusao$overall[[1]],
                            Kappa = matriz_confusao$overall[[2]],
                            Sensibilidade = matriz_confusao$byClass[[1]],
                            Especificidade = matriz_confusao$byClass[[2]],
                            PPV = matriz_confusao$byClass[[3]],
                            NPV = matriz_confusao$byClass[[4]],
                            Precisão = matriz_confusao$byClass[[5]],
                            Recall = matriz_confusao$byClass[[6]],
                            F1 = matriz_confusao$byClass[[7]],
                            AUC = auc
                            )
                            
  metricas = bind_rows(metricas, metricas_aux)
}

metricas

#e)Modelo final do melhor modelo escolhido a partir da métrica Acurácia
#Esse modelo deve ser aplicado ao restante da base de clientes a fim de indentificar os cleintes com maior potencial de 
#responder à camanpanha 
#Esse modelo final nos dá o Prediction, para achar a probabilidade de cada cliente aplicamos o prediction a Fórnula
#Probabilidade = e^Prediction/(1+e^Prediction)
#Se Probabilidade > 0.5 considera-se que o cliente vai aceitar a campanha.
ori_lreg$finalModel


