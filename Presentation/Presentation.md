# Application bancaire d'aide à l'octroi de prêt
Auteur: Pierre-Yves Prodhomme

## Objectif
Afin de pouvoir répondre à leur client de la manière la plus équitable/transparente possible, les instituts bancaires font apppel à du machine learning pour les aider dans l'octroi de prêt à leur client.
Ce notebook vise à décrire la méthode globale pour réaliser une application d'aide à l'octroi de prêt bancaire. 
*Ce projet est un POC qui n'inclut pas la gestion de la sécurité quand au transfert des données au travers du réseau.*
La méthode comprend différentes étapes qui correspondent au plan de cet article:
- collecte et préparation des données
- Sélection du modèle (benchmark)
    - Préprocess : réduction du dataset 
    - Stratégie et Préparation
    - Score et classement
- Rééquilibrage et augmentation des données
- déploiement de l'application
- Perspectives

## Résumé
Afin de limiter le nombre de données lors de l'évaluation des modèles, une  réduction du dataset (sub-dataset = 20% dataset initial) tout en conservant une distribution semblable au dataset de départ a été réalisée.
Les modèles sélectionnés ont ensuite été testés sur un datataset un peu plus large (40%) et un équilibrage des donnéesa a été expérimenté tout en continuant à optimiser les hyperparamètres du modèle choisi.
Finalement le modèle a été optimisé sur le dataset global légèrement modifié pour le rééquilibrage.
Enfin une application est mise en place dans le cloud, elle comprend une API déployée sur pythonanywhere.com avec flask et une UI sur streamlit.com.  
L'utilisateur (employé de banque) peut évaluer le risque de défaillance d'un candidat au prêt et connaître les raisons majeures (features importances) qui poussent à faire cette estimation.

## Collecte et Préparation des données
Le dataset de départ est récupéré sur [kaggle](https://www.kaggle.com/competitions/home-credit-default-risk).
Dans un premier temps le feature engineering est réalisé à partir du [notebook](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script ) de AGUIAR.

dataset relationship       | Data_engineering summary
:-------------------------:|:-------------------------:
![dataset_relationship](images/dataset_relationship.jpg) | ![data_engineering](images/data_engineering_reduced.jpg)

Le dataset de départ pour le benchmark des modèles contient donc un peu plus de 307 k.individus ayant 766 features.

## Sélection  du modèle
### Préprocess: réduction du dataset
Le dataset étant relativement large, afin d'accélérer les calculs nous constituons un dataset de plus petite dimension qui sera représentatif du dataset de départ. Chaque classe (classe 0 et classe 1) est traitée séparément.  
Le nombre de features est légèrement réduit via une ACP (les composantes principales représente 99% de la variance expliquée).  
Ensuite un clustering est réalisé toujours sur chaque classe séparément: chaque classe est réparti en 100 clusters.  
Ainsi il y a n_i échnatillons par cluster, on autorise un seul cluster avec moins de 10 individus.  
Ensuite pour constituer un sous-dataset, on pioche aléatoirement un ratio d'individus dans chaque cluster.  
Cela permet de constituer un dataset avec une répartition proche du dataset de départ.
  
![png](Presentation_files/Presentation_3_0.png)

![png](Presentation_files/Presentation_3_1.png)
    
### Stratégie et Préparation
Afin d'éviter de travailler sur de trop nombreuses variables en même temps, 
Le traitement de variables déséquilibrés est reporté à une phase ultérieure, lorsque un ou deux modèles se dégageront du lot.
La prise en compte d'un nombre importants de variables nécessiterait de multiplier les calculs, cela serait beaucoup trop chronophage.

**Réduction de dimension**  
D'autre part nous testons quelques réduction de dimension sur les features. D'un point de vue métier ce n'est pas pertinent, mais cela permet de vérifier si une réduction de dimension permettrait d'améliorer les scores obtenus. 

**Preprocess and Gridsearch**  
Un undesampling de 0.5 semble être raisonnable pour débuter le benchmark.  
Ainsi pour commencer notre benchmark des différents modèles, nous partons sur un pipeline:    
Pipeline(UnderSampling(0.5), StandardScaler(), Model) 

Les modèles avec les hyperparamètres testés sont listés ci-dessous dans le tableau.

<!-- Librairies  | sklearn |sklearn | sklearn | sklearn      | XGBoost | sklearn | LightGBM |  sklearn    | TF     -->  
<!--:-------------:|:-------:|:------:|:-------:|:------------:|:-------:|:-------:|:--------:|:-----------:|:------: -->
<!--**Modèles**    |  SVC    |  NBG   | LogReg  | RandomForest | XGBoost | HistGBM | LightGBM |    MLP      | MLP     -->

**Modèles**    |  Librairies | Hyperparamètres 
:-------------:|:-----------:|:----------:
Dummy          |  sklearn    |
SVC            |  sklearn    | {'C': [0.1, 1, 10], 'class_weight': ['balanced', None], 'gamma':[0.1, 1, 10] }
NaiveBayesian  |  sklearn    | Gaussian
kNN            |  sklearn    | {'n_neighbors': [5, 10, 20], 'leaf_size': [10, 30], 'weights':["uniform", "distance"] }
LogReg         |  sklearn    | {'penalty': ["l2", "none"], 'class_weight': ['balanced', None], 'max_iter':[100, 500] }
RandomForest   |  sklearn    | {'max_depth': [5, 10], "min_impurity_decrease": [0.01, 0.001],            "min_samples_leaf":[10, 20], "n_estimators":[100, 500], “class_weight":["balanced", None]}
XGBoost        |  XGBoost    | {'max_depth': [3, 6], 'n_estimators': [500, 1000], "learning_rate": [0.02, 0.005], "min_child_weight":[1, 10]} 
HistGBM        |  sklearn    | {'max_depth': [3, 6], 'max_iter': [500, 1000], "learning_rate": [0.1, 0.05], "min_samples_leaf":[20, 50]} 
LightGBM       |  LightGBM   | {'m__max_depth': [3, 6], "m__learning_rate": [0.1, 0.05], "m__min_child_samples":[20, 50]}
MLP (3 layers) |  sklearn    | {"dense_size" : [16, 12], "activation_hidden_layers" : ["tanh"], "class_weight": [None, "balanced"],  "m__dropout" : [0.5, 0.25, 0]
MLP (3 layers) |  TensorFlow | {"dense_size" : [16, 12], "activation_hidden_layers" : ["tanh"], "class_weight": [None, "balanced"],  "m__dropout" : [0.5, 0.25, 0]


### Score et classement
Le score utilisé pour le benchmark est roc_auc.  
**Avantage**: il permet d'estimer d'une manière globale la performance du modèle en précision ou en recall.  
**Inconvénient** : l'estimation est grossière et ne permet pas d'affiner le modèle pour un objectif précis.
Pour cela il vaut mieux utiliser le $F_\beta score$.  

Le graphe ci-dessous illustre la stabilité du modèle XGBOOST par rapport au jeu de données (axe des abscisses 10 split CV).  
Chaque courbe correspond à un jeu d'hyperparamètres.  
On affiche le score AUC obtenu par le set de test pour chaque split.
Sur le graphe, les 5 meilleurs score moyen sont affichés.  
Les scores moyens et les jeux de paramètres sont affichés dans le tableau suivant.
    
![png](Presentation_files/Presentation_6_0.png)
    
XGBOOST
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>param_m__learning_rate</th>
      <th>param_m__max_depth</th>
      <th>param_m__min_child_weight</th>
      <th>param_m__n_estimators</th>
      <th>mean_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.02</td>
      <td>3</td>
      <td>10</td>
      <td>500</td>
      <td>0.750310</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.02</td>
      <td>6</td>
      <td>10</td>
      <td>500</td>
      <td>0.750077</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.02</td>
      <td>3</td>
      <td>10</td>
      <td>1000</td>
      <td>0.749457</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.02</td>
      <td>3</td>
      <td>1</td>
      <td>500</td>
      <td>0.749059</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02</td>
      <td>3</td>
      <td>1</td>
      <td>1000</td>
      <td>0.747950</td>
    </tr>
  </tbody>
</table>
</div>



Cette vérification a été réalisée pour chaque modèle. Ci-dessous LightGBM.
  
![png](Presentation_files/Presentation_9_0.png)
    
Lightgbm
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>param_m__learning_rate</th>
      <th>param_m__max_depth</th>
      <th>param_m__min_child_samples</th>
      <th>mean_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.05</td>
      <td>3</td>
      <td>50</td>
      <td>0.746755</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.05</td>
      <td>3</td>
      <td>20</td>
      <td>0.744263</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.05</td>
      <td>6</td>
      <td>50</td>
      <td>0.740469</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.05</td>
      <td>6</td>
      <td>20</td>
      <td>0.738337</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.1</td>
      <td>6</td>
      <td>50</td>
      <td>0.736738</td>
    </tr>
  </tbody>
</table>
</div>

Ces graphes ne sont pas suffisant pour déterminer le meilleur model pour prédire une faillite d'un candidat.  
Une métrique $score_{final}$ est proposée pour permettre d'estimer le meilleur model:  
$\displaystyle score_{final} = score_{control} + \alpha\cdot diff_{score}+\frac{\beta}{t} $.  
Les coefficients $\alpha$ et $\beta$ ont été définis de la manière suivante :  
Les score control sont ordonnées et on ne retient que les 10 meilleurs.
Soit $\Delta$ la différence entre le $score_{control}$ max et le $score_{control}$ min, alors 
- la valeur obtenu pour le terme lié au temps d'exécution $\frac{\beta}{t}$ devait être au maximum égal à $\Delta$
- la valeur obtenue pour le terme lié à la différence de score $\alpha\cdot diff_{score}$ devait être au maximum égal à $\Delta$

Read file

![png](Presentation_files/Presentation_12_1.png)

**Modèle sélectionné**
Le modèle LightGBM est celui qui a obtenu le meilleur score final lors du benchmark.
Dans la suite nous nous concentrons donc sur ce modèle.

## Optimisation du modèle et correction du déséquilibre du jeu de données
Afin d'optimiser le modèle au plus proche de son utilisation final, il serait intéressant de comparer la perte du refus d'un prêt à un client viable avec  la perte subit lorsqu'un client est défaillant. Ne connaissant pas ce rapport, on part sur l'hypothèse qu'un client défaillant est 2 fois moins rentable qu'un client viable refusé (cette valeur est sûrement très sous estimée).
On choisit donc d'optimiser le $f_\beta$ score avec $\beta=2$, c'est à daire qu'on favorise le rappel 2 fois plus que la précision.

## Rééquilibrage des données
Une des problématique de ce type de sujet est que le jeu de données est déséquilibré et qu'il peut par conséquent engendrer un biais sur le modèle.  
ImbLearn est une librairie dédiée au rééquilibrage des données qui inclut différentes méthodes pour traiter ces données.
Nous tentons de corriger ce biais en utilisant différentes méthodes:
- Adasyn (Imb_learn)
- Undersampling (Imb_learn)
- TomekLinks (Imb_learn)
- NearMiss (Imb_learn)
- Smote (Imb_learn)

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model Name</th>
      <th>Hyperparameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LightGBM_optim2</td>
      <td>{'m__max_depth': [3], 'm__learning_rate': [0.1, 0.05, 0.01], 'm__min_child_samples': [200, 100, 50], 'm__n_estimators': [400], 'm__class_weight': ['balanced', None]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LightGBM_optim4</td>
      <td>{'m__max_depth': [3], 'm__learning_rate': [0.05], 'm__min_child_samples': [100], 'm__n_estimators': [400], 'm__class_weight': ['balanced', None], 'Ad__sampling_strategy': ['minority'], 'Ad__n_neighbors': [2, 5]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LightGBM_optim3</td>
      <td>{'m__max_depth': [3], 'm__learning_rate': [0.05], 'm__min_child_samples': [100], 'm__n_estimators': [400], 'm__class_weight': ['balanced', None], 'sm__sampling_strategy': [0.7, 0.3], 'sm__k_neighbors': [2, 5]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LightGBM_optim5</td>
      <td>{'m__max_depth': [3], 'm__learning_rate': [0.05], 'm__min_child_samples': [100], 'm__n_estimators': [400], 'm__class_weight': ['balanced', None], 'sm__sampling_strategy': [0.3], 'sm__k_neighbors': [5]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LightGBM_optim6</td>
      <td>{'m__max_depth': [3], 'm__learning_rate': [0.05], 'm__min_child_samples': [100], 'm__n_estimators': [400], 'm__class_weight': ['balanced', None], 'u__sampling_strategy': [0.1, 0.25], 'u__n_neighbors': [3, 5], 'sm__sampling_strategy': [0.3], 'sm__k_neighbors': [5]}</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LightGBM_optimFtwo_1</td>
      <td>{'m__max_depth': [3], 'm__learning_rate': [0.1, 0.05, 0.01], 'm__min_child_samples': [200, 100, 50], 'm__n_estimators': [400], 'm__class_weight': ['balanced', None]}</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LightGBM_optimFtwo_2</td>
      <td>{'m__max_depth': [3, 5], 'm__learning_rate': [0.05], 'm__min_child_samples': [200, 300], 'm__n_estimators': [400, 600], 'm__class_weight': ['balanced', None]}</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LightGBM_optimFtwo_3</td>
      <td>{'m__max_depth': [3, 5], 'm__learning_rate': [0.05], 'm__min_child_samples': [300], 'm__n_estimators': [400], 'm__class_weight': ['balanced'], 'sm__sampling_strategy': [0.6, 0.7, 0.8], 'u__sampling_strategy': [0.5]}</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LightGBM_optimFtwo_2u</td>
      <td>{'m__max_depth': [5], 'm__learning_rate': [0.05], 'm__min_child_samples': [300], 'm__n_estimators': [400], 'm__class_weight': ['balanced', None], 'u__sampling_strategy': [0.2, 0.5]}</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LightGBM_optimFtwo_2ub</td>
      <td>{'m__max_depth': [5], 'm__learning_rate': [0.05], 'm__min_child_samples': [300], 'm__n_estimators': [400], 'm__class_weight': ['balanced'], 'u__sampling_strategy': [0.3, 0.39999999999999997, 0.5, 0.6, 0.7, 0.7999999999999999, 0.8999999999999999, 1.0]}</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LightGBM_optimFtwo_4</td>
      <td>{'m__max_depth': [5, 6], 'm__learning_rate': [0.05], 'm__min_child_samples': [300, 400], 'm__n_estimators': [400, 500], 'm__class_weight': ['balanced'], 'u__sampling_strategy': [0.5, 0.6]}</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LightGBM_optimFtwo_5</td>
      <td>{'m__max_depth': [5], 'm__learning_rate': [0.05, 0.03], 'm__min_child_samples': [400], 'm__n_estimators': [500], 'm__class_weight': ['balanced'], 'sm__sampling_strategy': [0.55, 0.6], 'u__sampling_strategy': [0.5]}</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LightGBM_optimFtwo_6</td>
      <td>{'m__max_depth': [5, 6], 'm__learning_rate': [0.02, 0.03], 'm__min_child_samples': [400, 500], 'm__n_estimators': [500, 600], 'm__class_weight': ['balanced'], 'u__sampling_strategy': [0.5]}</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LightGBM_optimFtwo_7</td>
      <td>{'m__max_depth': [6], 'm__learning_rate': [0.03], 'm__min_child_samples': [400, 500], 'm__n_estimators': [600], 'm__class_weight': ['balanced'], 'u__sampling_strategy': [0.25]}</td>
    </tr>
  </tbody>
</table>
</div>

Adasyn vs. Smote

![png](Presentation_files/Presentation_17_1.png)
    
![png](Presentation_files/Presentation_17_3.png)

Undersampling    
![png](Presentation_files/Presentation_18_1.png)

Ci-dessus on voit que l'évolution du score avec l'undersampling effectué sur 40% du dataset n'est pas monotone.  
L'undersampling avec ratio=1 est le plus performant : le rappel est performant, mais la précision est dégradée.  

Le benchmark réalisé sur les différentes méthodes proposées par ImbLearn indique que :
- ces méthodes n'améliore pas nécessairement la performance (score), même en optimisant la méthode
- la performance de ces méthodes sont très sensible au dataset, donc l'application d'une méthode sur un sous-ensemble (avec une dstribution proche du dataset global) ne donnera pas nécessairement le même résultat que pour le dataset global.

## Optimisation du modèle et augmentation du jeu de données
L'augmentation de la taille de l'échantillon pour le training améliore le score.
Avec l'augmentation du jeu de données, on voit que :
- L'augmentation du nombre d'individus dans les feuilles tend à limiter le biais dans chaque feuille et donc améliore également le score (jusqu'à un certain point où on perd en précision), ci-dessous en passant de 300 à 500 échantillons par feuille on améliore les score.

![png](Presentation_files/Presentation_21_1.png)
    
- L'augmention de la profondeur permet de donner plus de DL au modèle, ce qui améliore le score (jusqu'à un certain point où le modèle sur-apprend). Ci-dessous, même si c'est peu flagrant, en augmentant la profondeur on améliore le modèle(calcul effectué sur 20% du dataset) 

![png](Presentation_files/Presentation_23_0.png)

- L'augmentation du nombre d'estimateurs tend à limiter le sur-apprentissage et donc améliore le score du modèle (jusqu'à un certain point où le modèle perd en précision), comme on peut le voir ci-dessous (apprentissage réalisé sur 40% du dataset)  
    
![png](Presentation_files/Presentation_25_1.png)
    
## Optimisation finale

Finalement, après divers ajustement pas à pas, le modèle évalué le plus performant correspond au paramètres suivants:  

**Hyperparamètres**|  Valeurs
:-----------------:|:-----------:
Undersample        |  0.5
max_depth          |  8
learning_rate      |   0.02
min_child_samples  |   600
n_estimators       |   700
class_weight       |   balanced

Le $F_\beta$ score ainsi obtenu est $F_{\beta=2}=0.452$, ce qui correspond à un $Recall=0.706$ et une $Precision=0.185$.
On voit que ce modèle est optimisé pour alerter l'utilisateur d'un risque sur un client. Mais dans ce cas un jugement humain devra prendre la décision afin de ne pas léser un trop grand nombre de candidat (mais cela nécessitera un financement supplémentaire de la part de la banque à prendre en compte pour optimiser au mieux le modèle).
L'optimisation du $F_\beta=2$ score  dégrade fortement la precision, l'AUC reste correct (autour de 0.786).

## Critique de la méthode de maximisation employée et perspectives

**Objectif**  
L'objectif de la méthode était d'optimiser les hyperparamètres en augmentant progressivement la taille du dataset afin de dégager une tendance en fonction de la taille du dataset.
Cela fonctionne "bien" si on fait varier les hyperparamètres unitairement, et on retrouve des tendances tel qu'illustrer ci-dessus. Mais un point que je n'avais pas anticipé, est que malgré la distribution semblable des sous-dataset construit, les tendances sur les hyperparmètres d'équilibrage sont très sensibles et ne suivent pas nécessairement les mêmes tendances. Cela a un peu désorienté les recherches.

**Inconvénients d'une méthode gridsearch**  
Une méthode privilégiée pour optimiser les hyperparamètres est le gridsearch.
Le problème c'est qu'il est nécessaire d'optimiser plusieurs hyperpamètres qui n'évoluent pas de manière monotone.
Si l'on veut évaluer systématiquement les hyperparamètres sur une grille on tombe rapidement sur une limite de ressource/temps.
Par exemple supposons que l'on souhaite évaluer 10 hyperparamètres, on target 5 valeurs (ce qui est peu) pour chaque hyperparamètre. Supposons que l'apprentissage soit de l'ordre de la seconde (sur le 20% du jeu de donnée) et que  on utilse 10 splits pour la CV, cela représente 1130 jours de calcul non-stop.
Et cette estimsation est une limite très basse, puisque le temps d'apprentissage est rarement de l'ordre de la seconde.

Et avec cette méthode nous n'explorons qu'une infime partie de l'espace des solutions.
2 choix s'offrent à nous
- augmenter les ressources de calcul (malheureusement pas de budget disponible pour ce projet)
- tester une autre méthode

**Méthode alternative employée**  
La méthode employée pour minimiser le score est l'évaluation du score sur des valeurs aléatoires d'hyperparamètres.
Cette méthode est aléatoire et a été réalisé à la main:
- d'une part c'est fastidieux
- d'autre part cela manque d'efficacité

**Perspectives**  
Il serait intéressant de mettre en place une méthode plus systématique/générique pour estimer sur une large zone les valeurs de score (méthode Monte Carlo ou RandomSearch), puis restreindre au fur et à mesure les zones de recherche.
Vérifier la tendance d'un paramètre "à la main" reste intéressant mais dans ce cas il vaut mieux faire varier 1 seul paramètre (voire 2) pour avoir une vision plus claire.

Enfin il serait intéressant de regarder du côté des librairies tel que Scikit-Optimize, d'une manière plus général ce [blog](https://neptune.ai/blog/hyperparameter-tuning-in-python-complete-guide) est une bonne entrée en matière.

## Features Importances/explicateur de modèle

![Shapley values](../Notebook/Shapley_Values_full.png)  
Les valeurs les plus importantes qui permettent d'expliquer le modèle sont représentées ci-dessus.
Les 3 premières features sont des données confidentiels appartenant à la banque. Les 2 premières sont clairement prépondérantes en comparaison des autres features Mais les prêts en cours, le taux de remboursement, l'âge et les autres features présentes sur le graphe influent aussi clairement sur les risques de défaillance évaluée par le modèle.

**Choix de l'explicateur de modèle**
Suite à une étude des différentes solutions proposées pour l'évaluation des features importances, le choix s'est porté sur SHAP.
Les méthodes proposées tel que la méthode features_importances disponible directement dans la librairie XGBoost ou LIME, permettent effectivement de faire ressortir l'importance de certains paramètres mais suivant certaines hypothèses qui peuvent être séduisante mais n'ont pas de fondement mathématiques et qui ne font pas concenus.
SHAP propose d'évalue l'importance d'une feature en calculant les Shapley values, c'est à dire en calculant l'impact que produit le retrait d'une feature sur le modèle en prenant en compte en compte les interractions avec les autres features (en calculant toutes les combinaisons possibles de retrait de features). C'est donc effectivement ce que l'on recherche.
Par ailleurs, SHAP est très rapide et optimisé pour les arbres de décisions tel que XGBOOST, LightGBM.
Il existe d'autres librairies mais pour le moment SHAP répond pleinement à mes exigences.

## Déploiement de la solution
Une app avec une interface utilisateur est proposé pour permettre d'évaluer le risque de défaillance d'un candidat.  
Cette app est déployée dans le cloud, l'architecture mis en place est présenté sur la figure suivante.  
![architecture_cloud](images/architecture_cloud.jpg)  
Le déploiement a été réalisé avec flask et streamlit.  
L'app est disponible sur le [lien](https://py53-projet7-dashboard-z498gi.streamlit.app/) (me contacter pour réactiver le serveur du model)

# Conclusion
La réduction du dataset en conservant une distribution proche du dataset de départ a permis de réaliser un benchmark assez large de différents modèles de classification.
L'optimisation du modèle est un sujet déclicat qui réclame de mettre en place des moyens et une méthodologie rigoureuse,
d'autant plus avec le déséquilibre des données inérant à ce type de projet.
La librairie Imblearn permet de traiter cette problématique de déséquilibre, mais permet pas de résoudre le problème qui ncessitera toujours d'aller rechercher des données supplémentaires.
Après optimisation sur l'ensemble des données, les SHAPley values permettent d'interpreter le modèle finalement
Le Déploiement d'une app permet de montrer le fonctionnement du modèle ainsi défini.
