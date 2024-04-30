# cardio-vasc-risks

## Présentation du projet

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
