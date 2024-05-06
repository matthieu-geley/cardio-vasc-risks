# cardio-vasc-risks

## Présentation du projet

Le but de ce projet est d'apprendre à utiliser la régression logistique dans le cadre de l'apprentissage supervisé.   
Pour celà nous allons utiliser un dataset contenant des informations médicales sur des patients anonymisés et nous allons essayer de prédire si ces patients sont à risque de maladies cardiovasculaires ou non.   


## Veille informatique sur la régression logistique

### Définition:

La régression logistique est un **modèle statistique** permettant d'étudier les relations entre un ensemble de **variables qualitatives (X)** et une **variable qualitative binaire (Y)**.   
Il s'agit d'un **modèle linéaire** généralisé utilisant une **fonction logistique** comme fonction de lien.   

Un modèle de régression logistique permet aussi de **prédire la probabilité** qu'un événement se produise (valeur de 1) ou non (valeur de 0) à partir de l'**optimisation des coefficients de régression**.   
Ce résultat varie toujours entre 0 et 1. Lorsque la valeur prédite est supérieure à un seuil (0.5 par défaut), l'événement est prédit comme se produisant. Sinon, il est prédit comme ne se produisant pas.

### Les hypothèses de la régression logistique:

1) La variable dépendante est binaire:   
La variable dépendante doit être classée en deux catégories distinctes. Cela signifie que la régression logistique prédit la probabilité d'un événement en deux scénarios possibles:   
- 1 si l'événement se produit
- 0 si l'événement ne se produit pas.

2) La distribution gaussienne:   
La régression logistique suppose que la relation entre les variables (entrée et sortie) est linéaire

3) Les variables indépendantes ne sont pas multicollinéaires:   
La régression logistique nécessite que les variables indépendantes ne soient pas fortement corrélées entre elles.

4) Un échantillon de grande taille:   
La régression logistique nécessite un échantillon de grande taille pour obtenir des résultats fiables.

### Mathématiquement:

Considérons un ensemble de données d'entrée X  definit comme suit:   
$$
X = {x_1, x_2, ..., x_n}
$$
La régression logistique a pour objectif de trouver une fonction h telle que nous puissions calculer:   

y = {1 si h(x) >= [seuil], 0 si h(x) < [seuil]}   

On comprend donc qu'on attend de notre fonction h qu'elle soit une probabilité comprise entre 0 et 1, paramétrée par = 1, 2, ..., n à optimiser et que le seuil que nous définissons correspond à notre critère de classification, généralement 0.5 .   

La fonction qui remplit le mieux ces conditions est la fonction sigmoïde, définit sur R à valeurs dans [0, 1]. Elle est définie comme suit:

$$
σ(x) = \frac{1}{1 + e^{-x}}
$$

Graphiquement, celle-ci correspond à une courbe en forme de S qui a pour limites 0 et 1 lorsque x tend vers -∞ et +∞ passant par y=0.5 lorsque x=0 .

![sigmoid function](images/sigmoidFunct.png)

### Pourquoi l'utiliser ?

La regression logistique est souvent utilisée lorsque la variable cible est catégorielle, faisant d'elle un outil de choix pour prédire des événements avec deux issues possibles tels que "oui" ou "non", "vrai" ou "faux", "positif" ou "négatif".
Elle est également utile quand la relation entre les variables indépendantes et dépendantes n'est pas linéaire.

### Différents type de régression logistiques ?

Il existe trois types de régressions logistiques:   
- **Binaire**: La variable dépendante a deux catégories.
- **Multinomiale**: La variable dépendante a trois catégories ou plus sans ordre.
- **Ordinale**: La variable dépendante a trois catégories ou plus avec un ordre.

1) **Régression logistique binaire**:   
La régression logistique binaire est utilisée pour prédire la relation entre deux variables:
- La variable dépendante (Y)
- La variable indépendante (X)

Exemple:   
- Oui ou non
- Vrai ou faux
- Positif ou négatif

2) **Régression logistique multinomiale**:
Dans le cas de la régression logistique multinomiale, nous avons une variable dépendante catégorielle avec trois catégories ou plus sans ordre.

3) **Régression logistique ordinale**:
Dans le cas de la régression logistique ordinale, nous avons une variable dépendante catégorielle avec trois catégories ou plus avec un ordre.   

Exemple:   
- faible, moyen, fort.
- d'accord, neutre, pas d'accord.

## Type de données de notre dataset
Données quantitatives discrètes:
- Age: âge en jours
- Gender: 1: femme, 2: homme
- Height: taille en cm
- Ap_hi: pression artérielle systolique
- Ap_lo: pression artérielle diastolique
- Cholesterol: 1: normal, 2: au-dessus de la normal, 3: bien au-dessus de la normal
- Glucose: 1: normal, 2: au-dessus de la normal, 3: bien au-dessus de la normal
- Smoke: 0: non-fumeur, 1: fumeur
- Alco: 0: ne boit pas, 1: boit
- Active: 0: pas d'activité physique, 1: activité physique
- Cardio: 0: pas de maladie cardiovasculaire, 1: maladie cardiovasculaire

Données quantitatives continues:
- Weight: poids en kg

## Penser à l'équité des données

### Qu'est ce que la standardisation ou normalisation des données ?

**La standardisation des données**, fait référence au processus de transformation des données brutes en une forme standardisée. La plupart du temps, cela implique de procéder à la modification des données afin que ces dernières obtiennent une **moyenne de zéro** et un **écart-type de 1**. En d'autres termes, la standardisation consiste à trier, organiser et homogénéiser des données suivant certains standards préalablement définis.

De cette manière, les erreurs de modélisation seront limitées et les ressources de l'entreprise seront optimisées, tandis que la comparaison de différentes variables sera grandement facilitée.

Toutefois, il faut garder à l'esprit qu'une standardisation de données parfaite et infaillible n'existe pas vraiment, la clé étant de procéder à l'analyse et à la gestion des données de manière régulière.

La **normalisation des données** est un processus couramment utilisé en statistiques, en science des données et en apprentissage automatique pour mettre à l'échelle les valeurs de différentes variables dans un même intervalle. L'objectif principal de la normalisation est de rendre les données comparables entre elles et plus facilement interprétables par les algorithmes d'analyse et de modélisation.

Les mots Normalisation et Standardisation sont souvent utilisés de manière à décrire un ensemble de méthodes permettant d'effectuer une mise à l'échelle, mais ils sont aussi utilisés pour décrire une technique bien précise de redimensionnement de variables. Nous utilisons normalisation pour décrire l'ensemble des méthodes: **Normalisation Min-Max**, **Standardisation (normalisation standard)** et **Robuste** 

### Les avantages de la standardisation :

Comme tout procédé informatique, la standardisation de données comporte de nombreux avantages pour les structures qui l'appliquent, dont les principaux sont :   
- **Une meilleur comparaison des données** : leur standardisation favorise la comparaison de différents types de variables sur une même échelle de mesure, ce qui joue également un rôle déterminant dans leur compréhension et leur exploitation.
- **Une réduction des erreurs de modélisation** : l'élimination des redondances, des mauvaises labellisations ou encore des informations obsolètes garantissant la qualité et la pertinence des modèles et rapports produits.
- **Une amélioration de la performance des algorithmes utilisés** : la réduction des écarts de valeurs entre les différents type de données permet de significativement améliorer la convergence ainsi que le fonctionnement général de certains algorithmes, en particulier lorsqu'il s'agit d'algorithmes de Machine Learning.
- **Une optimisation des ressorces technologiques de l'entreprise** : grâce à la suppression de l'ensemble des données doublons, inutiles ou oblsolètes, la standardisation des données conduit à un gain plus ou moins important d'espaces de stockage sans oublier une augmentation de la vitesse de traitement des données en général. 

#### Les étapes préalables à la standardisation des données :

Avant de procéder à la standardisation des données, il est nécessaire de suivre un certain nombre d'étapes préalables, dont les principales sont :
- **L'accessibilité des données** : tous les utilisateurs doivent être en mesure d'accéder facilement aux données dont ils ont besoin ainsi que parvenir à les comprendre et les exploiter sans difficultés et sans risques d'erreurs d'intérprétation, conformément aux standards de l'entreprise.
- **La mise à jour régulière des référentiels** : afain d'assurer leur exactitude et de préserver leur plus-value, les référentiels de données doivent être régulièrement actualisés et non rester figés.
- **L'implication de l'ensemble des acteurs de l'entreprise** : chaque segment de l'entreprise doit être mobilisé afin que les bases de données standardisées répondent bel et bien à leurs besoins.

#### Les étapes à suivre pour standardiser des données :

Afin de pleinement <ins>réussir la standardisation de vos données</ins>, il est crucial de suivre un certain nombre d'étapes clés, la première d'entre elles étant reliée à l'identification précides des données devant faire l'objet de cette standardisation.

##### Étape n°1 : Identifier les données à standardiser

La première étape essentielle d'une standardisation des données efficace consiste à correctement ifentifier les données devant être traitées. En ce sens, il est recommandé de **les stocker dans un emplacement dédié** tels q'une base de données particulière, un fichier CSV ou tout autre support ou format de fichier jugé adéquat.

##### Étape n°2 : Calculer la moyenne et l'écart-type

La deuxième étape : la détermination de la moyenne et de l'écart-type nécessaire au processus de standardisation. Le plus simple consiste à utiliser un **logiciel d'analyse de données** spécialement conçu à cet effet.

Pour calculer cette <ins>moyenne</ins>, il faut additionner toutes les valeurs de données, puis diviser cette somme par le nombre total de valeur dans l'ensemble de données.

Exemple de données : $$ x_1, x_2, x_3, ..., x_n $$
La formule de caclul de la moyenne correspond ainsi à : $$ moyenne = \frac{x_1 + x_2 + x_3 + ... + x_n}{n} $$

Pour sa part, l'écart-type permet de mesurer la dispersion des valeurs dans l'ensemble de données par rapport à la moyenne obtenue. Dès lors, il se calcule en :
- Soustrayant la moyenne de chaque valeur de l'ensemble de données,
- Élevant le résultat au carré,
- Additionnant l'ensemble de ces carrés,
- Divisant cette somme par le nombre total de valeurs dans l'ensemble de données,
- Moins 1 à ce résultat,
- Puis enfin, on prend la racine carrée de cette somme pour aboutir à l'écart-type.

##### Étape n°3 : Débuter le processus de standardisation

À partir de la moyenne et de l'écart-type pré-déterminés, la standardisation peut être lancée en soustrayant la moyenne de l'ensemble des valeurs puis en divisant le résultat par l'écart-type.

Le principal intérêt de cette opération est de pouvoir ramener toutes les données à une échelle commune ainsi que de les exprimer en unités d'écart-type. En effet, une fois que des données ont été standardisées, elles ont une **moyenne de 0** et un **écart-type de 1**, ce qui signifie que toutes les observations sont exprimées en unités d'écart-type par rapport à la moyenne de la population. Cette standardisation facilite donc l'analyse et la modélisation des données en permettant une comparaison plus facile des différentes variables.

##### Étape n°4 : Vérifier la standardisation des données

Dès lors que l'ensemble des étapes précédentes ont été dûment accomplies, la dernière étape consiste à tester la réussite du processus de standardisation, c'est-à-dire d'en vérifier les résultats. Pour cela, vous pouvez notamment vérifier que la moyenne des données standardisées est bien égale à zéro, ou encore que leur écart-type est égal à un. Si tel n'est pas le cas, il est alors nécessaire de revoir les calculs effectués ou bien de directement vérifier l'exactitude des données d'origine.

Source: [yzr.ai: Comment standardiser des données ?](https://www.yzr.ai/articles/comment-standardiser-des-donnees/#:~:text=Le%20plus%20simple%20consiste%20%C3%A0,dans%20l'ensemble%20de%20donn%C3%A9es.)

### La normalisation des données :

#### Pourquoi normaliser les données ?

Dans de nombreux cas, les données peuvent avoir des échelles très différentes, c'est-à-dire que certaines variables peuvent avoir des valeurs beaucoup plus grandes ou plus petites que d'autres. Cela peut poser  des problèmes pour certaines techniques statistiques ou algorithmes d'apprentissage automatique, car ils peuvent être sensibles à l'échelle des données. La normalisation permet de résoudre ce problème en ajustant les valeurs des variables pour qu'elles se situent dans un intervalle spécifié, souvent entre 0 et 1, ou autour de la moyenne avec un écart-type donné.

#### Les avantages associé à la normalisation des données :

La normalisation des données améliore la qualité, la performance et l'interprétabilité des analyse statistiques et des modèles d'apprentissage automatique en éliminant les problèmes liés à l'échelle des variables, et en permettant une comparaison plus juste entre différentes caractéristiques des données. Dans les faits, cela se traduit par des avantages concrets tels que :

- **Comparabilité maximale** : les données normalisées sont mises à la même échelle, permettant une comparaison et une interprétation plus facile entre différentes variables.

- **Optimisation de l'apprentissage automatique** : la normalisation facilite la convergence plus rapide des algorithmes d'apprentissage automatique en réduisant l'échelle des variables, aidant ainsi à atteindre des résultats fiables et consolidés plus rapidement.

- **Stabilité renforcée des modèles** : la nprmalisation réduit l'impact des valeurs extrêmes (outliers) et rend les modèles plus stables et résistant aux variations des données.

- **Amélioration de l'interprétabilité** : la normalisation des données facilite l'interprétation des coefficients, rendant l'analyse plus compréhensible.

#### Les différentes méthodes de normalisation des données :

Il existe plusieurs méthodes de normalisation des données, chacune ayant ses propres avantages et inconvénients. Les méthodes les plus couramment utilisées sont :

- **Min-Max Scaling** : cette méthode repose sur le principe d'une mise à l'échelle des valeurs d'une variable afin que celles-ci se situent dans un intervalle spécifié, généralement entre 0 et 1. Cette technique est particulièrement utile lorsque l'on souhaite conserver la relation linéaire entre les valeurs originales.

  Formule: $$ X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}} $$
  Où:
  - X est la valeur d'origine
  - X_min est la valeur minimale de la variable
  - X_max est la valeur maximale de la variable

- **Z-Score Normalization** ([ref. standardisation](#étape-n1--identifier-les-données-à-standardiser)) : est une technique qui répond davantage à un impératif de standardisation. Cette méthode consiste à ajuster les valeurs d'une variable pour qu'elles aient une moyenne de 0 et un écart-type de 1. Contrairement à la méthode Min-Max Scaling, la standardisation n'impose pas de limite supérieure ou inférieure spécifique aux valeurs transformées. Cette technique est recommandée lorsque les variables ont des échelles très différentes, car elle permet de centrer les données autour de zéro et de les mettre à la même échelle par rapport à l'écart-type.

  Formule: $$ X_{norm} = \frac{X - \mu}{\sigma} $$
  Où:
  - X est la valeur d'origine
  - μ est la moyenne des valeurs de la variable
  - σ est l'écart-type des valeurs de la variable

D'autres méthodes peuvent également être envisagées dans une optique de normalisation des données, mais elles sont plus marginales. Retenons tout de même la normalisation par décimale (**Decimal Scaling**) ou la normalisation par vecteur unitaire (**Unit Vector Scaling**)

- **Decimal Scaling** : cette méthode consiste à diviser chaque valeur d'ube variable par une puissance de 10 en fonction du nombre de chiffres significatifs. Cela déplace la virgule vers la gauche, plaçant le chiffre le plus significatif à gauche de la virgule. Cette technique ajuste les valeurs pour qu'elles se situent dans un intervalle plus petit, simplifiant ainsi les calculs.

- **Unit Vector Scaling** : cette méthode consiste à diviser chaque valeur d'un vecteur de données par la norme euclidienne du vecteur, transformant ainsi le vecteur en un vecteur unitaire (de longueur 1). Cette technique est souvent employée dans des algorithmes qui calculent les distances ou les similarités entre des vecteurs.

#### Différence entre normalisation et standardisation :

La normalisation et la standardisation des données répondent au même enjeu de représentativité de la donné mais dans des perspectives différentes. Bien qu'elles soient toutes les deux des techniques de mise à l'échelle des données, elles diffèrent dans la manière dont elles transforment les valeurs des variables.

La standardisation transforme les valeurs d'une variable pour qu'elles aient une moyenne de 0 et un écart-type de 1. Contrairement à la normalisation, la standardisation ne fixe pas de plage spécifique pour les valeurs transformées. La standardisation est utile lorsque les variables ont des échelles très différentes, et elles permet de centrer les données autour de zéro et de les mettre à l'échelle par rapport à l'écart-type, ce qui peut faciliter l'interprétation des coefficients dans certains modèles. En fonction de la nature des données et des enseignements que l'on souhaite en tirer, il faudra tantôt recourir à la normalisation, tantôt à la standardisation.

Source: [Zeenea.com: Qu'est-ce que la normalisation des données ?](https://zeenea.com/fr/quest-ce-que-la-normalisation-des-donnees/)

### La méthode Robuste :

La méthode Robuste est une technique de normalisation des données qui est moins sensible aux valeurs aberrantes (outliers) que la méthode Min-Max Scaling ou Z-Score Normalization. Le procédé est le suivant: on soustrait aux valeurs de la variable la médiane et on divise par l'écart interquartile (IQR). L'écart interquartile est la différence entre le troisième quartile (Q3) et le premier quartile (Q1) de la distribution des valeurs de la variable.

Formule: $$ X_{rob} = \frac{X - \text{médiane}}{\text{IQR}} $$
Où:
- X est la valeur d'origine
- médiane est la médiane des valeurs de la variable
- IQR est l'écart interquartile des valeurs de la variable:
$$ IQR = Q3 - Q1 $$
Où:
- Q3 est le troisième quartile (75 % des valeurs sont inférieures à Q3)
- Q1 est le premier quartile (25 % des valeurs sont inférieures à Q1)

Source: [Alliage-ad.com: Les méthodes de normalisation avec Scikit-Learn](https://www.alliage-ad.com/tutoriels-python/les-methodes-de-normalisation/)