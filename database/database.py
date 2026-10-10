"""Affiche une fiche de contact choisie depuis le terminal."""


CONTACTS = {
    "1": "Sofiane Ait Ameur — Saint-Geneviève-des-Bois (91700)",
    "2": "Rabah — Saint-Geneviève-des-Bois, à la frontière de Fleury-Mérogis",
    "3": "Wylen Belaidi — Saint-Geneviève-des-Bois (91700), Saint-Hubert",
}


def main():
    print("Choisis une fiche : 1, 2 ou 3")
    print("0. Quitter")

    while True:
        choix = input("\nEntre un numéro (0 pour quitter) : ").strip()
        if choix == "0":
            print("Au revoir !")
            break
        if choix in CONTACTS:
            print(CONTACTS[choix])
        else:
            print("Numéro invalide. Choisis 1, 2, 3 ou 0.")


if __name__ == "__main__":
    main()