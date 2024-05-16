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
- 0 si l'événement ne se produit pas.
- 1 si l'événement se produit.

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

Graphiquement, celle-ci correspond à une courbe en forme de S qui a pour limites 0 et 1 lorsque x tend  vers -∞ et +∞ passant par y=0.5 lorsque x=0.

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

La standardisation et la normalisation des données sont des méthodes de réduction de l'échelle des variables pour les rendre comparables.   
Ces méthodes sont couramment utilisées en statistiques, en science des données et en apprentissage automatique pour mettre à l'échelle les valeurs de différentes variables dans un même intervalle.   

Les mots Normalisation et Standardisation sont souvent utilisés de manière à décrire un ensemble de méthodes permettant d'effectuer une mise à l'échelle, mais ils sont aussi utilisés pour décrire une technique bien précise de redimensionnement de variables.   

**La normalisation = "Min-Max scaling"   
La standardisation = "Z-score normalisation"**

### Les avantages :

Comme tout procédé informatique, la standardisation de données comporte de nombreux avantages pour les structures qui l'appliquent, dont les principaux sont :   
- **Une meilleur comparaison des données** : leur standardisation favorise la comparaison de différents types de variables sur une même échelle de mesure, ce qui joue également un rôle déterminant dans leur compréhension et leur exploitation.
- **Une réduction des erreurs de modélisation** : l'élimination des redondances, des mauvaises labellisations ou encore des informations obsolètes garantissant la qualité et la pertinence des modèles et rapports produits.
- **Une amélioration de la performance des algorithmes utilisés** : la réduction des écarts de valeurs entre les différents type de données permet de significativement améliorer la convergence ainsi que le fonctionnement général de certains algorithmes, en particulier lorsqu'il s'agit d'algorithmes de Machine Learning.
- **Une optimisation des ressorces technologiques de l'entreprise** : grâce à la suppression de l'ensemble des données doublons, inutiles ou oblsolètes, la standardisation des données conduit à un gain plus ou moins important d'espaces de stockage sans oublier une augmentation de la vitesse de traitement des données en général. 

#### La normalisation :

La normalisation permet de résoudre les problèmes de dimensions en ajustant les valeurs des variables pour qu'elles se situent dans un intervalle spécifié, souvent entre 0 et 1, ou autour de la moyenne avec un écart-type donné.   
Pour cela il faut soustraire les valeurs par le minimum puis diviser par le maximum de toutes les observations.

Formule:   
$$ X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}} $$
  Où:
  - X est la valeur d'origine
  - X_min est la valeur minimale de la variable
  - X_max est la valeur maximale de la variable

#### La standardisation :

**La standardisation des données**, fait référence au processus de transformation des données brutes en une forme standardisée. La plupart du temps, cela implique de procéder à la modification des données afin que ces dernières obtiennent une **moyenne de zéro** et un **écart-type de 1**. En d'autres termes, la standardisation consiste à trier, organiser et homogénéiser des données suivant certains standards préalablement définis.

Formule:   
$$ X_{norm} = \frac{X - \mu}{\sigma} $$
  Où:
  - X est la valeur d'origine
  - μ (mu) est la moyenne des valeurs de la variable
  - σ (sigma) est l'écart-type des valeurs de la variable

#### D'autres méthodes :

##### La méthode Robuste :

La méthode Robuste est une technique de normalisation des données qui est **moins sensible** aux **valeurs aberrantes** (outliers) que la méthode Min-Max Scaling ou Z-Score Normalization.   
Le procédé est le suivant: on soustrait aux valeurs de la variable la médiane et on divise par l'écart interquartile (IQR). L'écart interquartile est la différence entre le troisième quartile (Q3) et le premier quartile (Q1) de la distribution des valeurs de la variable.

Formule:   
$$ X_{rob} = \frac{X - \text{médiane}}{\text{IQR}} $$
Où:
- X est la valeur d'origine
- médiane est la médiane des valeurs de la variable
- IQR est l'écart interquartile des valeurs de la variable:
$$ IQR = Q3 - Q1 $$
Où:
- Q3 est le troisième quartile (75 % des valeurs sont inférieures à Q3)
- Q1 est le premier quartile (25 % des valeurs sont inférieures à Q1)

##### La méthode de normalisation par décimale :

La méthode **Decimal Scaling** consiste à diviser chaque valeur d'ube variable par une puissance de 10 en fonction du nombre de chiffres significatifs. Cela déplace la virgule vers la gauche, plaçant le chiffre le plus significatif à gauche de la virgule. Cette technique ajuste les valeurs pour qu'elles se situent dans un intervalle plus petit, simplifiant ainsi les calculs.

##### La méthode de normalisation par Vecteur Unitaire :

La méthode **Unit Vector Scaling** consiste à diviser chaque valeur d'un vecteur de données par la norme euclidienne du vecteur, transformant ainsi le vecteur en un vecteur unitaire (de longueur 1). Cette technique est souvent employée dans des algorithmes qui calculent les distances ou les similarités entre des vecteurs.

#### Choisir entre la normalisation et la standardisation :

Le choix entre la standardisation et la normalisation dépend de la distribution des données et de l'algorithme d'apprentissage automatique que vous utilisez. Voici quelques conseils pour vous aider à choisir entre la standardisation et la normalisation :

- **Standardisation** : utilisez la standardisation lorsque les données suivent une distribution normale ou gaussienne ou lorsqu'il y a des données abérrantes.

- **Normalisation** : Pour les réseaux de neurones, on préfèra la normalisation car elle permet de réduire le temps de convergence. Pour les algorithmes qui ne sont pas sensibles à l'échelle des variables, la normalisation peut être une bonne option.

Sources:   
[yzr.ai: Comment standardiser des données ?](https://www.yzr.ai/articles/comment-standardiser-des-donnees/#:~:text=Le%20plus%20simple%20consiste%20%C3%A0,dans%20l'ensemble%20de%20donn%C3%A9es.)   
[Zeenea.com: Qu'est-ce que la normalisation des données ?](https://zeenea.com/fr/quest-ce-que-la-normalisation-des-donnees/)   
[Alliage-ad.com: Les méthodes de normalisation avec Scikit-Learn](https://www.alliage-ad.com/tutoriels-python/les-methodes-de-normalisation/)

## Utilisation de l'imc et du poids ?

La **multicollinéarité** est un potentiel problème dans la régression logistique.   

Mais qu'est ce que c'est ?   
La multicollinéarité est une situation dans laquelle deux variables (ou plus) **indépendantes** dans un modèle de régression sont **fortement liées entre elles**.   
Or les variables **indépendantes** ne doivent pas **dépendre** d'autres variable (sinon, elles ne sont plus **indépendantes**).

<u>Exemple:</u>
- une variable dépendante: le prix d'une maison
- des variables indépendantes: la taille de la maison, le nombre de chambres, le nombre de salles de bain

Si la taille de la maison augmente, le prix augmente aussi.
Il n'y as pas de multicolinéarité car la taille n'augmente pas intrinsèquement le nombre de chambres ou de salles de bain.

En revanche, si une nouvelle variable indépendante: taille des chambres est une mesure reprenant le nombre de chambre et la taille de la maison, alors il y a une multicolinéarité.

Dans notre cas, l'IMC est une combinaison du poids et de la taille.
Il y a donc une multicolinéarité entre l'IMC et le poids.

Il y existe deux type de multicolinéarité:
- la multicolinéarité structurel: n'est pas présente dans le jeu de donné initiale mais est créée par la transformation des données (comme l'IMC).
- la multicolinéarité de données: est présente dans le jeu de donnée initiale.

### Quel problèmes peut engendré la multicolinéarité ?

Les coefficients deviennent très sensible au moindre petit changement dans le modèle.
Réduit la précision des coefficients d'estimation ce qui réduit la puissance statistiques du modèle de régression.

Plus la multicolinéarité est forte, plus la problématique est importante.
Cependant, ces problèmes n'affectent que les variables indépendantes corrélé.

On peut par exemple avoir un modèle avec de forte multicolinéarité et une bonne prédiction car les variables corrélées ne sont pas des variables importantes pour la prédiction.

### Doit-on corriger la multicolinéarité ?

Le besoin de réduire la multicolinéarité dépend de:
- la force de la multicolinéarité
- l'objectif du modèle

### L'utilisation des VIF (Variance Inflation Factor) pour détecter la multicolinéarité:

Le VIF identifie le degré de la corrélation entre les variables indépendantes et la force de cette corrélation.

Le VIF commence à 1 et n'as pas de limite supérieur.
1 à 5: Modéré, pas correction nécessaires
supérieur à 5: Sévère, Coefficient et p-value non fiables, correction nécessaire

### Correction de la multicolinéarité:

- multicolinéarité structurel: centré les variables continues

- multicolinéarité de données: supprimer certaines variables hautement corrélés, combiner linéairement les variables indépendantes corrélées, utiliser LASSO([Least Absolute Shrinkage and Selection Operator par Robert Tibshirani en 1996](https://www.xlstat.com/fr/solutions/fonctionnalites/regression-lasso)) ou [Ridge regression](https://www.ibm.com/topics/ridge-regression)

Dans notre cas, nous choisissons de ne pas utiliser le poids au profit de l'IMC pour éviter la multicolinéarité.

