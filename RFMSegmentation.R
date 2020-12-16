#1. IMPORTAR BIBLIOTECAS NECESSÁRIAS PARA AS ANÁLISES
library(tidyverse)
library(gtsummary)
library(writexl)
library(ca)
library(ggplot2)
library(reshape2)
library(pracma)
library(lubridate)
#devtools::install_github("thomasp85/patchwork")
library(patchwork)
library(corrplot)
library(AppliedPredictiveModeling)
library(caret)
library(e1071)
# selected output
library(dplyr)
library(tidyr)

#2. LER O CONJUNTO DE DADOS DIRETAMENTE DA FONTE INDICADA - NESSE CASO DIRETO DO GITHUB
data <- read_csv("https://github.com/ifood/ifood-data-advanced-analytics-test/raw/master/ml_project1_data.csv", locale(encoding = "ISO-8859-1"),col_names = TRUE,col_types = NULL )

#3. INDICAR O CAMINHO ATIVO PARA SALVAR OS ARQUIVOS GERADOS DURANTE A EXECUÇÃO DO cÓDIGO
path <- "C:\\Users\\janai\\Documents\\Food"
setwd(path)

#4.FAZER A SEGMENTAÇÃO DOS CLIENTES
#Optei por fazer a segmentação dos clientes com base na técnica RFM - Recência, Frequência e Monetariedade.
#Nesse caso, vou fazer uma adaptação e em vez da frequência de compras vou usar a frequiência com que o cliente aceitou oferta em alguma campanha prévia.
#Em relação a Monetariedade, vou somar o volume gasto em cada tipo de Produto.
#A escolha dessa técnica foi influenciada pela Ranking gerado na questão 1 das váriaveis mais importantes a variável de interesse Response.

#a) Criar feature de interesse

#Criar feature QtdeCmp a partir das Features AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5 
# Essa Feature soma quantas campanhas o clinete aceitou antes da última

#Criar feature SumMnt a partir das Features MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProducts
# Essa Feature soma o volume de gasto em cada tipo de produto

data$QtdeCmp <- NA
data$SumMnt <- NA


for (n in 1:nrow(data)) {
  data$QtdeCmp[n] <- data$AcceptedCmp1[n] + data$AcceptedCmp2[n] + data$AcceptedCmp3[n] + data$AcceptedCmp4[n] + data$AcceptedCmp5[n]
  data$SumMnt[n] <- data$MntWines[n] + data$MntFruits[n] + data$MntMeatProducts[n] + data$MntFishProducts[n] + data$MntSweetProducts[n] + data$MntGoldProds[n]
}


#b) Criar campos para contabilizar o score de cada variável de interesse e preenchê-los
#Para cada variável usada na segmentação deve ser atribuído um score de 1 a 4. 
#Para as variáveis Recency e SumMnt o score é atribuído de acordo com o qaurtil. Para a variável QtdeCmp o score é 1 se QtdeCmp = 0 ou 1 e se QtdeCmp = 4 ou 5 score ´4 caso contrário é o próprio valor da variável. 
#Ainda, para a segmentação é preciso calcular a médica dos scores das variáveis SumMnt e QtdeCmp.

data$ScoreSumMnt <- NA
data$ScoreQtdeCmp <- NA
data$ScoreRecency <- NA

#Estabelecer os quintis
Q1_SumMnt <- quantile(data$SumMnt, probs = 0.25)
Q2_SumMnt <- quantile(data$SumMnt, probs = 0.5)
Q3_SumMnt <- quantile(data$SumMnt, probs = 0.75)

Q1_Recency <- quantile(data$Recency, probs = 0.25)
Q2_Recency <- quantile(data$Recency, probs = 0.5)
Q3_Recency <- quantile(data$Recency, probs = 0.75)


for (n in 1:nrow(data)) {
  
  if (data$QtdeCmp[n] <= 1){
    data$ScoreQtdeCmp[n] <- 1
  }  else if (data$QtdeCmp[n] >=4){
    data$ScoreQtdeCmp[n] <- 4
  }  else{
    data$ScoreQtdeCmp[n] <- data$QtdeCmp[n]
  }
  
  if (data$SumMnt[n] <= Q1_SumMnt){
    data$ScoreSumMnt[n] <- 1
  }else if (data$SumMnt[n] > Q1_SumMnt && data$SumMnt[n] <= Q2_SumMnt) {
    data$ScoreSumMnt[n] <- 2
  }else if (data$SumMnt[n] > Q2_SumMnt && data$SumMnt[n] <= Q3_SumMnt) {
    data$ScoreSumMnt[n] <- 3
  }else{
    data$ScoreSumMnt[n] <- 4
  }
  
  
  if (data$Recency[n] <= Q1_Recency){
    data$ScoreRecency[n] <- 1
  }else if (data$Recency[n] > Q1_Recency && data$Recency[n] <= Q2_Recency) {
    data$ScoreRecency[n] <- 2
  }else if (data$Recency[n] > Q2_Recency && data$Recency[n] <= Q3_Recency) {
    data$ScoreRecency[n] <- 3
  }else{
    data$ScoreRecency[n] <- 4
  }
  
}

#c)Segmentar os clientes nas categoris usando os ScoreRecency e o AvgScore_QtdeSum
#Core: RFM Socre = 111 >>> Highly engaged customers who have bought the most recent, the most often, and generated the most revenue.
#Loyal: RFM Socre = X12 >>> Customers who buy the most often from your company
#Whales: RFM Socre = XX1 >>> Customers who have generated the most revenue for your 
#Promising: RFM Socre = X13 X14 >>> Customers who return often, but do not spend a lot.
#Rookiens: RFM Socre = 14X >>> First time buyers on your site.
#Slipping: RFM Socre = 44X >>> Great past customers who haven't bought in awhile.

data$Segmento <- NA

for (n in 1:nrow(data)) {
  
  if (data$ScoreRecency[n] == 1 && data$ScoreQtdeCmp[n] ==1 && data$ScoreSumMnt[n] ==1){
    data$Segmento[n] <- "Core"
  } else if (data$ScoreRecency[n] <= 4 && data$ScoreQtdeCmp[n] ==1 && data$ScoreSumMnt[n] ==2){
    data$Segmento[n] <- "Loyal"
  } else if (data$ScoreRecency[n] > 1 && data$ScoreQtdeCmp[n] >1 && data$ScoreSumMnt[n] ==1){
    data$Segmento[n] <- "Whales"
  }  else if (data$ScoreRecency[n] <=4 && data$ScoreQtdeCmp[n] ==1 && data$ScoreSumMnt[n] >=3){
    data$Segmento[n] <- "Promising"
  } else if (data$ScoreRecency[n] ==1 && data$ScoreQtdeCmp[n] ==4 && data$ScoreSumMnt[n] <=4){
    data$Segmento[n] <- "Rookies"
  }   else{
    data$Segmento[n] <- "Slipping"
  }
  
}


#d)Visualizar quantidade de clientes por segmento

data = data %>%
   mutate(Segmento = factor(Segmento, levels = c("Core", "Loyal", "Whales",
                                                 "Promising", "Rookies", "Slipping"))) 

ggplot(data = data) + 
    geom_bar(mapping = aes(data$Segmento, fill = data$Segmento))+
    theme(legend.position = 'none') +
    labs(x = "", y = "Number of customers")  

Segmento_Response <- data %>% group_by(Response, `Segmento`)  %>%  tally()
Segmento_Response

#Criar função para fazer os gráficos de maneira mais fácil

grafico_boxplot <- function (x,y){
  
  ggplot(data) + 
    geom_boxplot(aes(x = x, y = y, fill = x)) + 
    theme_bw(15) +
    labs(x = label_x, y = label_y) + 
    theme(legend.position = 'none')
  
}

#Análise dos produtos que cada segmento costuma mais comprar

label_x <- "Customer Segmentation"
label_y <- "Spent on Wines (US$)"
p1<- grafico_boxplot(data$Segmento, data$MntWines)

label_x <- "Customer Segmentation"
label_y <- "Spent on Fruits (US$)"
p2<- grafico_boxplot(data$Segmento, data$MntFruits)

label_x <- "Customer Segmentation"
label_y <- "Spent on Meat Products (US$)"
p3<- grafico_boxplot(data$Segmento, data$MntMeatProducts)

label_x <- "Customer Segmentation"
label_y <- "Spent on Fish Products (US$)"
p4<- grafico_boxplot(data$Segmento, data$MntFishProducts)

label_x <- "Customer Segmentation"
label_y <- "Spent on Sweet Products (US$)"
p5<- grafico_boxplot(data$Segmento, data$MntSweetProducts)


label_x <- "Customer Segmentation"
label_y <- "Spent on Gold Products (US$)"
p6<- grafico_boxplot(data$Segmento, data$MntGoldProds)

p1 + p2 + p3 + p4 + p5 + p6

#Análise dos canis de compra que cada segmento costuma usar

label_x <- ""
label_y <- "Purchases from Web"
p7<- grafico_boxplot(data$Segmento, data$NumWebPurchases)

label_x <- ""
label_y <- "Purchases from Catalog"
p8<- grafico_boxplot(data$Segmento, data$NumCatalogPurchases)

label_x <- ""
label_y <- "Purchases from Store"
p9<- grafico_boxplot(data$Segmento, data$NumStorePurchases)

label_x <- ""
label_y <- "Purchases with discount"
p10<- grafico_boxplot(data$Segmento, data$NumDealsPurchases)


p7 + p8 + p9 + p10













