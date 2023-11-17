import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from get_data import get_training_data


FILTERS = 8


def create_model(filters):
    """Create a sequential NN for training on our images"""
    model = Sequential(
        [
            # une couche de neurones convolutifs input shape correspond aux dimensions des images
            Conv2D(
                filters,
                (3, 3),
                input_shape=(28, 28, 1),
                activation="relu",
                padding="same",
            ),
            # une couche de MaxPool2D
            MaxPool2D(3, 3),
            # une autre couche de neurones convolutifs et son MaxPool
            Conv2D(filters, 3, activation="relu", padding="same"),
            MaxPool2D(3, 3),
            # Couche Flatten
            # on met tout à plat à la fin
            Flatten(),
            # Couche dropout
            # on ignore 20% des neurones au hasard. Cela évite à notre réseau d'être
            # trop spécialisé sur son jeu d'entraînement
            Dropout(0.2),
            # la couche finale, avec 10 neurones, correspondant aux 10 valeurs possibles pour une image
            Dense(10, activation="softmax"),
        ]
    )
    ### compilation : ajout des paramètres de fonction de perte, d'optimiseur, et le critère d'évaluation (précision)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def main():
    working_directory = os.getcwd()
    ## Data loading
    X, y = get_training_data()

    # add data augmentation by rotations
    # X, y = get_augmented_data(X, y)

    ## data val split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=0.7,
    )

    ## training parameters
    epochs = 20
    patience = 5
    earlystop = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    ## model instance
    model = create_model(filters=FILTERS)
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,  # le nombre d'images traitées dans une vague
        epochs=epochs,  # nombre d'époques au maximum
        validation_data=(
            X_val,
            y_val,
        ),  # le jeu de données sur lesquels les tests de validation sont faits.
        verbose=1,  # contrôle la quantité de texte affiché pendant l'entraînement
        callbacks=[earlystop],
    )

    accuracy = max(history.history["val_accuracy"])
    percent_format = "{:.1%}"
    print(f"Modèle entraîné ! Précision estimée à {percent_format.format(accuracy)}.")

    os.makedirs(os.path.join(working_directory, "checkpoints"), exist_ok=True)

    # Save the weights
    weight_path = os.path.join(working_directory, "checkpoints/my_checkpoint")
    model.save_weights(weight_path)
    print("Poids du modèle enregistrés pour utilisation future.")


if __name__ == "__main__":
    main()
