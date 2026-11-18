<p align="center">
  <a href="README.md">English</a>
  &nbsp;·&nbsp;
  <a href="README.fr.md"><b>Français</b></a>
  &nbsp;·&nbsp;
  <a href="README.es.md">Español</a>
</p>

Espace latent partagé entre la parole et le texte. La recherche texte vers audio est la direction forte. L’audio vers le texte passe par une tête CTC (environ 55 % de WER en éwé dans les notes du dépôt). L’anglais LibriSpeech sert de langue de validation. Une démo playlist peut associer une requête texte à un morceau.

Le dépôt contient déjà `infer.py`, `eval.py`, `eval_ctc.py`, `app_playlist.py` et `run_playlist_app.sh`. Ce script shell vise des dossiers fixes du cluster du projet (`/home/data/projets-aps/projet6`). Utilisez la section Usage de son `README.md` pour la commande qui correspond à votre machine. Installez d’abord les paquets de `requirements.txt`.


La documentation anglaise déjà présente dans ce dépôt reste dans le fichier anglais.
