import numpy as np
import matplotlib.pyplot as plt


def show_image(position, ax, X, y):
    """prend une position et un objet matplotlib.Axes
    qu'on appelle ax et affiche sur cet ax l'image située à la position indiquée dans X,
    avec le libellé correspondant pris dans y."""
    # on va chercher une ligne et on la convertit en array
    img = np.array(X.loc[position])
    # on la reshape en 28x28 et on l'affiche sur l'ax avec imshow
    ax.imshow(img.reshape(28, 28), cmap="gray")
    # on va chercher dans y_train à la même position le libellé de l'image (le nombre écrit)
    ax.set_xlabel(y.loc[position])
    # on masque les graduations sur les axes des x et des y
    ax.set_xticks([])
    ax.set_yticks([])


def show_grid(n_rows, n_cols, X, y):
    """crée une grille d'Axes avec le nombre indiqué de lignes et de colonnes,
    et appelle la fonction show_image() sur chaque ax pour y afficher les premiers nombres de notre jeu de données
    """
    # on crée une figure et une grille d'ax appelées axs
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    # on itère sur chaque ax, on y appelle la fonction précédente show_image,
    # enumerate nous donne le numéro de l'ax, axs.flat permet de traverser la grille "à plat"
    for i, ax in enumerate(axs.flat):
        show_image(i, ax, X, y)
