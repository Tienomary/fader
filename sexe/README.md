README
Description du Projet

Ce projet vise à entraîner et évaluer des modèles d'apprentissage automatique pour des tâches liées à la classification du sexe à partir d'images ou de données spécifiques. Le projet comprend des scripts pour le prétraitement des données, l'entraînement des modèles, l'évaluation des performances et la génération de visualisations.
Structure des Dossiers et Fichiers
Répertoires

    __pycache__/ : Contient les fichiers compilés Python (*.pyc) pour l'exécution plus rapide des scripts.
    train_sexe20epoch/ : Répertoire contenant les résultats des entraînements du modèle sur 20 époques.
    train_sexe50epoch/ : Répertoire contenant les résultats des entraînements du modèle sur 50 époques.
    train_sexe70epoch/ : Répertoire contenant les résultats des entraînements du modèle sur 70 époques.

Fichiers Principaux

    README.md : Ce fichier fournit des explications sur la structure du projet et l'utilisation des scripts.
    after_female.png : Image illustrant les résultats après une transformation spécifique liée à la classe "female".
    before.png : Image avant toute modification ou transformation.
    courbe.py : Script Python générant les courbes de performances (précision, perte, etc.) lors de l'entraînement du modèle.
    data.py : Script pour le chargement et le prétraitement des données d'entraînement.
    model.py : Définition des architectures de modèle utilisées dans le projet (autoencodeur, discriminateur, etc.).
    test_model.py : Script permettant de tester les modèles entraînés sur des données de validation ou test.
    testautoencoder.py : Script spécifique pour tester la qualité de reconstruction du modèle autoencodeur.
    train_adversarial.py : Script pour l'entraînement d'un modèle dans un cadre d'apprentissage antagoniste (adversarial learning).
    train_disc.py : Script pour entraîner un discriminateur dans le cadre d'une approche adversariale.
    training.py : Script général pour l'entraînement du modèle principal, incluant des paramètres tels que le nombre d'époques.

Instructions d'Utilisation

    Préparation des Données :
    Exécutez le fichier data.py pour charger et préparer les données d'entraînement.

    Entraînement des Modèles :
        Pour entraîner un autoencodeur, utilisez train_adversarial.py.
        Pour entraîner un discriminateur, exécutez train_disc.py.
        Pour un entraînement standard, utilisez training.py.

    Visualisation des Courbes :
    Le fichier courbe.py permet de tracer les courbes de perte et de précision à partir des données générées lors de l'entraînement.

    Évaluation des Modèles :
    Utilisez test_model.py ou testautoencoder.py pour tester la performance des modèles entraînés sur un ensemble de test.

Notes Supplémentaires

    Assurez-vous que toutes les dépendances nécessaires (TensorFlow, PyTorch, numpy, matplotlib, etc.) sont installées.
    Les résultats des différents entraînements (train_sexe20epoch, train_sexe50epoch, train_sexe70epoch) permettent de comparer la performance du modèle sur différents nombres d'époques.
