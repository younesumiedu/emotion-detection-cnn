# Détection des Émotions Faciales avec CNN

**Projet de Fin d'Études (PFE)**  
Licence Sciences Mathématiques et Informatique (SMI)  
Faculté Polydisciplinaire de Ouarzazate – Université Ibn Zohr  

Réalisé par : Soulaiman BOUALI & Younes MOURABIT  
Encadré par : Prof. Othmane BAIZ  
Année universitaire : 2023/2024  

---

## Résumé du Projet

Dans ce projet, nous avons conçu et développé un système intelligent capable de détecter et classifier les émotions humaines à partir d’images ou en temps réel via une webcam.

Ce travail s’inscrit dans le domaine de l’intelligence artificielle et plus précisément du Deep Learning. Nous avons utilisé les réseaux de neurones convolutifs (CNN), reconnus pour leur efficacité dans les tâches de vision par ordinateur.

L’objectif principal est de proposer une solution permettant d’identifier l’état émotionnel d’une personne à partir des caractéristiques de son visage, afin de contribuer à des applications telles que la sécurité, la surveillance ou encore le suivi médical.

---

## Objectifs

- Développer un modèle capable de détecter les émotions faciales  
- Implémenter un système de reconnaissance en temps réel  
- Exploiter les performances des réseaux de neurones convolutifs (CNN)  
- Évaluer les performances du modèle à l’aide de métriques (accuracy, loss)  

---

##  Méthodologie

Nous avons adopté une approche basée sur les réseaux de neurones convolutifs (CNN) en raison de leur capacité à :

- Extraire automatiquement les caractéristiques des images  
- Gérer efficacement les données visuelles  
- Réduire le nombre de paramètres par rapport aux réseaux classiques  

###  Architecture du modèle

Le modèle CNN utilisé se compose des couches suivantes :

- **Couche de convolution (CONV)** : extraction des caractéristiques  
- **Fonction d’activation (ReLU)** : introduction de non-linéarité  
- **Couche de pooling (POOL)** : réduction de dimension  
- **Couche Flatten** : transformation en vecteur  
- **Couche Fully Connected (FC)** : classification  
- **Fonction Softmax** : calcul des probabilités des classes  

---

## Données utilisées

Nous avons utilisé un ensemble de données composé d’environ :

-  35 000 images  
-  Images de taille 28x28  
- 7 classes d’émotions  

Répartition des données :

- 90% pour l’entraînement  
- 10% pour le test  

---

## Technologies et Outils

- **Langage de programmation :** Python  
- **Deep Learning :** TensorFlow, Keras  
- **Vision par ordinateur :** OpenCV  
- **Interface graphique :** Tkinter  
- **Analyse et visualisation :** NumPy, Matplotlib  
- **Environnement de développement :** Anaconda, Jupyter Notebook, VS Code  

