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

#e)Criação da Feautre Count apenas para servir de contador em algumas análises
data$Count <- 1

#5. FAZER ANÁLISE EXPLORATÓRIA DOS DADOS
#Análise por Estatística Descritica (Inferência) e Análise por Visualização de Dados

#5.1 ANÁLISE POR ESTATÍSTICA DESCRITIVA - INFERÊNCIA

#5.1.1 Fazer Tabela Descritiva
data_descritiva <- data

#a)Verificar o Summary para analisar tipos de dados
summary(data_descritiva)

#b)Transformar os Dados para o Tipo Desejado
data_descritiva$Education <- as.factor(data_descritiva$Education)
data_descritiva$Marital_Status <- as.factor(data_descritiva$Marital_Status)
data_descritiva$Kidhome <- as.factor(data_descritiva$Kidhome)
data_descritiva$Teenhome <- as.factor(data_descritiva$Teenhome)
data_descritiva$AcceptedCmp1 <- as.factor(data_descritiva$AcceptedCmp1)
data_descritiva$AcceptedCmp2 <- as.factor(data_descritiva$AcceptedCmp2)
data_descritiva$AcceptedCmp3 <- as.factor(data_descritiva$AcceptedCmp3)
data_descritiva$AcceptedCmp4 <- as.factor(data_descritiva$AcceptedCmp4)
data_descritiva$AcceptedCmp5 <- as.factor(data_descritiva$AcceptedCmp5)
data_descritiva$Complain <- as.factor(data_descritiva$Complain)
data_descritiva$Response <- as.factor(data_descritiva$Response)
data_descritiva$TimeEnrollment <- as.numeric(data_descritiva$TimeEnrollment)
data_descritiva$SomeCmp <- as.factor(data_descritiva$SomeCmp)

#c)Verificar o Summary novamente para verificar se transformação foi bem sucedida
summary(data_descritiva)

#d)Remover outliers
remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 3 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}

data_descritiva$Age <- remove_outliers(data_descritiva$Age)
data_descritiva$Income <- remove_outliers(data_descritiva$Income)


#e) Modificar os dados para que representem os textos desejados
#Assumi a premissa que os Status Alone, YOLO e Absurd podem ser agregados numa categoria Other uma vez que apresentam poucos registros e não sçao uma categoria comum
colunas = c("AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "SomeCmp",
            "Complain")
colunas2 = c("Marital_Status")
colunas3 = c("Response")

data_descritiva = data_descritiva %>%
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
                                                  x == "0" ~ "Not Accepted")}) %>%
  mutate(Response = factor(Response, levels = c("Not Accepted", "Accepted"))) 
  

#e)Selecionar Colunas desejadas e define número de casas decimais
data_descritiva2 <- data_descritiva 
 
casas_decimais = 2

#f) Mudar nomes dos rótulos das variáveis
lista_labels = list(Count ~ "Total, No (%)",
                    Age ~ "Age (years), median (IQR)",
                    Education ~ "Education level, No (%)",
                    Marital_Status ~ "Marital Status, No (%)",
                    Income ~ "Income (US$), median (IQR)",
                    TimeEnrollment ~ "Enrollment time (days), median (IQR)",
                    Kidhome ~ "Number Kids at Home, No (%)",
                    Teenhome ~ "Number of Teenagers at Home, No (%)",
                    Recency ~ "Time since last purchase (days), median (IQR)",
                    MntWines ~ "Spent on Wine (US$), median (IQR)",
                    MntFruits ~ "Spent on Fruits (US$), median (IQR)",
                    MntMeatProducts ~ "Spent on Meat Products (US$), median (IQR)",
                    MntFishProducts ~ "Spent on Fish Products (US$), median (IQR)",
                    MntSweetProducts ~ "Spent on Sweet Products (US$), median (IQR)",
                    MntGoldProds ~ "Spent on Gold Products (US$), median (IQR)",
                    NumDealsPurchases ~ "Purchases with discount, median (IQR)",
                    NumWebPurchases ~ "Purchases from Web, median (IQR)",
                    NumCatalogPurchases ~ "Purchases from catalog, median (IQR)",
                    NumStorePurchases ~ "Purchases from store, median (IQR)",
                    NumWebVisitsMonth ~ "Web visits (per month), median (IQR)",
                    AcceptedCmp1 ~ "Accepted offer 1º campaing, No (%)",
                    AcceptedCmp2 ~ "Accepted offer 2º campaing, No (%)",
                    AcceptedCmp3 ~ "Accepted offer 3º campaing, No (%)",
                    AcceptedCmp4 ~ "Accepted offer 4º campaing, No (%)",
                    AcceptedCmp5 ~ "Accepted offer 5º campaing, No (%)",
                    SomeCmp ~ "Accepted offer in some campaing, No (%)",
                    QtdeCmp ~ "Number of offers accepted in previous campaing, median (IQR)",
                    Complain ~ "Costumer complain, No (%)")

#g)Criar tabela descritiva e exportar para uma arquivo Excel para o caminho definido anteriormente
table1 = data_descritiva2 %>% 
  tbl_summary(by = Response,
              label = lista_labels,
              missing = "no",
              statistic = list(Count ~ "{n}")) %>% 
  add_overall() %>% 
  add_n()
table1

descritiva = table1$table_body %>% 
    select(-c(variable, row_type, n)) %>% 
    rename("Response" = "label",
           "Total" = "stat_0",
           "Not Accepted" = "stat_1",
           "Accepted" = "stat_2")
descritiva
write_xlsx(descritiva, "tabela_descritiva_responses.xlsx")


#5.1.2 Fazer matriz de correlação das variáveis numéricas
#a) Selecionar apenas as colunas numéricas. Para essa análise as variáveis KidHome, TeenHome, Response, SomeCmp, Complain vão ser consideradas numéricas
data_numericas =
  data %>%
  dplyr::select(Age, Income, TimeEnrollment, Recency, MntWines, MntFruits, MntMeatProducts, MntFishProducts, 
         MntSweetProducts, MntGoldProds, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, 
         NumWebVisitsMonth, QtdeCmp, Kidhome, Teenhome, Complain, Response, SomeCmp)
data_numericas$TimeEnrollment <- as.numeric(data_numericas$TimeEnrollment)
summary(data_numericas)

#b)Desconsiderar os NAs para essa análise
data_numericas <- na.omit(data_numericas)

#c) Gerar a Matriz de Correlação
corrplot(cor(data_numericas), method = "circle")
#obs: quanto maior o círculo maior a correlação entre as variaveis.
#Além disso, quanto mais azul escuro, mais próxima a correlação fica de 1, que significa que além de forte a correlação é positiva. 
#Equivalentemente quanto mais próximo de vermelho escuro, mais próxima a correlação fica de -1, que significa que além de forte a 
#correlação é negativa.


#5.2 ANÁLISE POR VISUALIZAÇÃO DE DADOS - GRÁFICOS
#A partir da análise inferencial é possível perceber algumas relações entre variáveis que ajudam a guiar
#o desenvolvimento de alguns gráficos
#Com esses gráficos tem-se o objetivo de encontrar mais insights que ainda não foram vistos nas análises anteriores


  #a)Criar função para fazer os gráficos de maneira mais fácil
  grafico_dispersao_numericas <- function (x,y){
    
    ggplot(data = data_descritiva) + 
      geom_point(mapping = aes(x = x, y = y))+
      geom_smooth(aes(y = y, x = x))+
      facet_wrap(~Response)+
      labs(x = label_x, y = label_y)
    
  }
  
  grafico_boxplot <- function (x,y){
    
    ggplot(data_descritiva) + 
      geom_boxplot(aes(x = x, y = y, fill = x)) + 
      theme_bw(15) +
      labs(x = label_x, y = label_y) + 
      facet_wrap(~ Response) +
      theme(legend.position = 'none')
    
  }
  
#b) Criar os gráficos de interesse usando as funções
  
 ######################################Análise de Renda
label_x <- "Age (years)"
label_y <- "Income (US$)"
p1 <- grafico_dispersao_numericas(data_descritiva$Age, data_descritiva$Income)
p1

label_x <- "Spent on Wine (US$)"
label_y <- "Income (US$)"
p2 <- grafico_dispersao_numericas(data_descritiva$MntWines, data_descritiva$Income)

label_x <- "Spent on Fruits (US$)"
label_y <- "Income (US$)"
p3 <- grafico_dispersao_numericas(data_descritiva$MntFruits, data_descritiva$Income)

label_x <- "Spent on Meat Products (US$)"
label_y <- "Income (US$)"
p4 <- grafico_dispersao_numericas(data_descritiva$MntMeatProducts, data_descritiva$Income)

label_x <- "Spent on Fish Products (US$)"
label_y <- "Income (US$)"
p5 <- grafico_dispersao_numericas(data_descritiva$MntFishProducts, data_descritiva$Income)

label_x <- "Spent on Sweet Products (US$)"
label_y <- "Income (US$)"
p6 <- grafico_dispersao_numericas(data_descritiva$MntSweetProducts, data_descritiva$Income)

label_x <- "Spent on Gold Products (US$)"
label_y <- "Income (US$)"
p7 <- grafico_dispersao_numericas(data_descritiva$MntGoldProds, data_descritiva$Income)

p2 + p3 + p4 + p5 + p6 + p7

label_x <- "Purchases with discount"
label_y <- "Income (US$)"
p8 <- grafico_dispersao_numericas(data_descritiva$NumDealsPurchases, data_descritiva$Income)
p8

label_x <- "Purchases from Web"
label_y <- "Income (US$)"
p9<- grafico_dispersao_numericas(data_descritiva$NumWebPurchases, data_descritiva$Income)

label_x <- "Purchases from Catalog"
label_y <- "Income (US$)"
p10<- grafico_dispersao_numericas(data_descritiva$NumCatalogPurchases, data_descritiva$Income)

label_x <- "Purchases from Store"
label_y <- "Income (US$)"
p11<- grafico_dispersao_numericas(data_descritiva$NumStorePurchases, data_descritiva$Income)

p9 + p10 + p11

label_x <- "Number of Kids in home"
label_y <- "Income (US$)"
p12<- grafico_boxplot(data_descritiva$Kidhome, data_descritiva$Income)

label_x <- "Number of Teenagers in home"
label_y <- "Income (US$)"
p13<- grafico_boxplot(data_descritiva$Teenhome, data_descritiva$Income)

p12 + p13

label_x <- "Marital Status"
label_y <- "Income (US$)"
p14<- grafico_boxplot(data_descritiva$Marital_Status, data_descritiva$Income)
p14

label_x <- "Accepted offer in some campaing"
label_y <- "Income (US$)"
p15<- grafico_boxplot(data_descritiva$SomeCmp, data_descritiva$Income)
p15

######################################Análises Canais de Compra x Gastos
label_x <- "Purchases from Web"
label_y <- "Spent on Wine (US$)"
p16<- grafico_dispersao_numericas(data_descritiva$NumWebPurchases, data_descritiva$MntWines)

label_x <- "Purchases from Catalog"
label_y <- "Spent on Wine (US$)"
p17<- grafico_dispersao_numericas(data_descritiva$NumCatalogPurchases, data_descritiva$MntWines)

label_x <- "Purchases from Store"
label_y <- "Spent on Wine (US$)"
p18<- grafico_dispersao_numericas(data_descritiva$NumStorePurchases, data_descritiva$MntWines)


label_x <- "Purchases from Web"
label_y <- "Spent on Fruits (US$)"
p19<- grafico_dispersao_numericas(data_descritiva$NumWebPurchases, data_descritiva$MntFruits)

label_x <- "Purchases from Catalog"
label_y <- "Spent on Fruits (US$)"
p20<- grafico_dispersao_numericas(data_descritiva$NumCatalogPurchases, data_descritiva$MntFruits)

label_x <- "Purchases from Store"
label_y <- "Spent on Fruits (US$)"
p21<- grafico_dispersao_numericas(data_descritiva$NumStorePurchases, data_descritiva$MntFruits)

label_x <- "Purchases from Web"
label_y <- "Spent on Meat Products (US$)"
p22<- grafico_dispersao_numericas(data_descritiva$NumWebPurchases, data_descritiva$MntMeatProducts)

label_x <- "Purchases from Catalog"
label_y <- "Spent on Meat Products (US$)"
p23<- grafico_dispersao_numericas(data_descritiva$NumCatalogPurchases, data_descritiva$MntMeatProducts)

label_x <- "Purchases from Store"
label_y <- "Spent on Meat Products (US$)"
p24<- grafico_dispersao_numericas(data_descritiva$NumStorePurchases, data_descritiva$MntMeatProducts)

label_x <- "Purchases from Web"
label_y <- "Spent on Fish Products (US$)"
p25<- grafico_dispersao_numericas(data_descritiva$NumWebPurchases, data_descritiva$MntFishProducts)

label_x <- "Purchases from Catalog"
label_y <- "Spent on Fish Products (US$)"
p26<- grafico_dispersao_numericas(data_descritiva$NumCatalogPurchases, data_descritiva$MntFishProducts)

label_x <- "Purchases from Store"
label_y <- "Spent on Fish Products (US$)"
p27<- grafico_dispersao_numericas(data_descritiva$NumStorePurchases, data_descritiva$MntFishProducts)

label_x <- "Purchases from Web"
label_y <- "Spent on Sweet Products (US$)"
p28<- grafico_dispersao_numericas(data_descritiva$NumWebPurchases, data_descritiva$MntSweetProducts)

label_x <- "Purchases from Catalog"
label_y <- "Spent on Sweet Products (US$)"
p29<- grafico_dispersao_numericas(data_descritiva$NumCatalogPurchases, data_descritiva$MntSweetProducts)

label_x <- "Purchases from Store"
label_y <- "Spent on Sweet Products (US$)"
p30<- grafico_dispersao_numericas(data_descritiva$NumStorePurchases, data_descritiva$MntSweetProducts)

label_x <- "Purchases from Web"
label_y <- "Spent on Gold Products (US$)"
p31<- grafico_dispersao_numericas(data_descritiva$NumWebPurchases, data_descritiva$MntGoldProds)

label_x <- "Purchases from Catalog"
label_y <- "Spent on Gold Products (US$)"
p32<- grafico_dispersao_numericas(data_descritiva$NumCatalogPurchases, data_descritiva$MntGoldProds)

label_x <- "Purchases from Store"
label_y <- "Spent on Gold Products (US$)"
p33<- grafico_dispersao_numericas(data_descritiva$NumStorePurchases, data_descritiva$MntGoldProds)

p16 + p17 + p18 + p19 + p20 + p21 + p22 + p23 + p24 + p25 + p26 + p27 + p28 + p29 + p30 + p31 + p32 + p33


######################################Análise Volume de gastos e compras com desconto
label_x <- "Purchases with discount"
label_y <- "Spent on Wine (US$)"
p34<- grafico_dispersao_numericas(data_descritiva$NumDealsPurchases, data_descritiva$MntWines)

label_x <- "Purchases with discount"
label_y <- "Spent on Fruits (US$)"
p35<- grafico_dispersao_numericas(data_descritiva$NumDealsPurchases, data_descritiva$MntFruits)

label_x <- "Purchases with discount"
label_y <- "Spent on Meat Products (US$)"
p36<- grafico_dispersao_numericas(data_descritiva$NumDealsPurchases, data_descritiva$MntMeatProducts)

label_x <- "Purchases with discount"
label_y <- "Spent on Fish Products (US$)"
p37<- grafico_dispersao_numericas(data_descritiva$NumDealsPurchases, data_descritiva$MntFishProducts)

label_x <- "Purchases with discount"
label_y <- "Spent on Sweet Products (US$)"
p38<- grafico_dispersao_numericas(data_descritiva$NumDealsPurchases, data_descritiva$MntSweetProducts)

label_x <- "Purchases with discount"
label_y <- "Spent on Gold Products (US$)"
p39<- grafico_dispersao_numericas(data_descritiva$NumDealsPurchases, data_descritiva$MntGoldProds)

p34 + p35 + p36 + p37 + p38 + p39

######################################Análise Volume de gastos e Resposta à alguma campanha prévia
label_x <- "Accepted offer in some campaing"
label_y <- "Spent on Wine (US$)"
p40<- grafico_boxplot(data_descritiva$SomeCmp, data_descritiva$MntWines)

label_x <- "Accepted offer in some campaing"
label_y <- "Spent on Fruits (US$)"
p41<- grafico_boxplot(data_descritiva$SomeCmp, data_descritiva$MntFruits)

label_x <- "Accepted offer in some campaing"
label_y <- "Spent on Meat Products (US$)"
p42<- grafico_boxplot(data_descritiva$SomeCmp, data_descritiva$MntMeatProducts)

label_x <- "Accepted offer in some campaing"
label_y <- "Spent on Fish Products (US$)"
p43<- grafico_boxplot(data_descritiva$SomeCmp, data_descritiva$MntFishProducts)

label_x <- "Accepted offer in some campaing"
label_y <- "Spent on Sweet Products (US$)"
p44<- grafico_boxplot(data_descritiva$SomeCmp, data_descritiva$MntSweetProducts)

label_x <- "Accepted offer in some campaing"
label_y <- "Spent on Gold Products (US$)"
p45<- grafico_boxplot(data_descritiva$SomeCmp, data_descritiva$MntGoldProds)

p40 + p41 + p42 + p43 + p44 + p45

######################################Análise da Variável Kidhome x Volume de Gastos
label_x <- "Number of Kids at home"
label_y <- "Spent on Wine (US$)"
p46<- grafico_boxplot(data_descritiva$Kidhome, data_descritiva$MntWines)

label_x <- "Number of Kids at home"
label_y <- "Spent on Fruits (US$)"
p47<- grafico_boxplot(data_descritiva$Kidhome, data_descritiva$MntFruits)

label_x <- "Number of Kids at home"
label_y <- "Spent on Meat Products (US$)"
p48<- grafico_boxplot(data_descritiva$Kidhome, data_descritiva$MntMeatProducts)

label_x <- "Number of Kids at home"
label_y <- "Spent on Fish Products (US$)"
p49<- grafico_boxplot(data_descritiva$Kidhome, data_descritiva$MntFishProducts)

label_x <- "Number of Kids at home"
label_y <- "Spent on Sweet Products (US$)"
p50<- grafico_boxplot(data_descritiva$Kidhome, data_descritiva$MntSweetProducts)

label_x <- "Number of Teenagers at home"
label_y <- "Spent on Gold Products (US$)"
p51<- grafico_boxplot(data_descritiva$Kidhome, data_descritiva$MntGoldProds)

p46 + p47 + p48 + p49 + p50 + p51

######################################Análise da Variável Kidhome x Canais de compra
label_x <- "Number of Kids at home"
label_y <- "Purchase from Web"
p52<- grafico_boxplot(data_descritiva$Kidhome, data_descritiva$NumWebPurchases)

label_x <- "Number of Kids at home"
label_y <- "Purchase from Catalog"
p53<- grafico_boxplot(data_descritiva$Kidhome, data_descritiva$NumCatalogPurchases)

label_x <- "Number of Kids at home"
label_y <- "Purchases from Store"
p54<- grafico_boxplot(data_descritiva$Kidhome, data_descritiva$NumStorePurchases)

label_x <- "Number of Teenagers at home"
label_y <- "Purchase from Web"
p52_2<- grafico_boxplot(data_descritiva$Teenhome, data_descritiva$NumWebPurchases)

label_x <- "Number of Teenagers at home"
label_y <- "Purchase from Catalog"
p53_2<- grafico_boxplot(data_descritiva$Teenhome, data_descritiva$NumCatalogPurchases)

label_x <- "Number of Teenagers at home"
label_y <- "Purchases from Store"
p54_2<- grafico_boxplot(data_descritiva$Teenhome, data_descritiva$NumStorePurchases)

p52 + p53 + p54 + p52_2 + p53_2 + p54_2

######################################Análise da Variável Kidhome e Teenhome x Canais de compra
label_x <- "Number of Kids at home"
label_y <- "Purchase with discount"
p55<- grafico_boxplot(data_descritiva$Kidhome, data_descritiva$NumDealsPurchases)

label_x <- "Number of Teenagers at home"
label_y <- "Purchase with discount"
p56<- grafico_boxplot(data_descritiva$Teenhome, data_descritiva$NumDealsPurchases)

p55 + p56


#5.3 ANÁLISE DA IMPORTÂNCIA DAS VARIÁVEIS PARA O CLIENTE ACEITAR A OFERTA NA CAMPANHA
#A análise de importância das variáveis acessa a importância relativa das variáveis usando o valor absoluto da
#estatística t. Com isso, é possível formar um ranking das variáveis mais importantes, ou seja, que mais influenciam
#na definição da classificação da viagem nesse caso específico
#PAra isso, usei um modelo simples de regressão logística.

#a)Retirar registros com dados faltantes. Como nessa base foram poucos casos resolvi remover, mas caso contrário seria necessário fazer um imputação.
data_descritiva3 <- na.omit(data_descritiva)

#b) Altera dados da variável de interesse para um formato aceitável para o modelo
data_descritiva3 = data_descritiva3 %>%
  mutate_at(vars(colunas3), function(x){ case_when(x == "Accepted" ~ "Accepted", 
                                                   x == "Not Accepted" ~ "Not.Accepted")}) %>%
  mutate(Response = factor(Response, levels = c("Accepted", "Not.Accepted"))) 


#c) Criar o conjunto de treino e teste
set.seed(2 ^ 31 - 1)
trainIndex <- createDataPartition(data_descritiva3$Response, p = .8, 
                                  list = FALSE, 
                                  times = 1)
dados_treino<-data_descritiva3[trainIndex,]
dados_teste<-data_descritiva3[-trainIndex,]

#d) Definie como será feito o treino do modelo com 10-fold cross validation
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  classProbs = TRUE,
  repeats = 5,
  savePredictions = TRUE
)


#e) Rodar o modelo
lreg<-train(Response~Age+ Education+ Marital_Status+ Income + TimeEnrollment+ Kidhome+ Teenhome +Recency+ MntWines+ MntFruits+ MntMeatProducts+ MntFishProducts+
            MntSweetProducts+ MntGoldProds+ NumDealsPurchases+ NumWebPurchases+ NumCatalogPurchases+ NumStorePurchases+ NumWebVisitsMonth+ AcceptedCmp1+
            AcceptedCmp2+ AcceptedCmp3+ AcceptedCmp4+ AcceptedCmp5+ SomeCmp +QtdeCmp+ Complain,
            data=dados_treino,
            method="glm",
            family = binomial(),
            trControl = fitControl)

#f)Pegar informações sobre o modelo
summary(lreg)

#g) Pegar ranking de importância das variáveis
varImp(lreg)


