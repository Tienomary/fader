import kagglehub
# Kaggle code, faut mettre les datas dans le dossiers datas après qu'elles soient download
# l'architecture finale de datas est : 
# datas / img_align_celeba / img_align_celeba / liste des images
# data / 4 fichiers .csv donc attributs et partitions 
# Download latest version
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

print("Path to dataset files:", path)