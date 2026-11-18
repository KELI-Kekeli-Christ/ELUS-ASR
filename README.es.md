<p align="center">
  <a href="README.md">English</a>
  &nbsp;·&nbsp;
  <a href="README.fr.md">Français</a>
  &nbsp;·&nbsp;
  <a href="README.es.md"><b>Español</b></a>
</p>

Espacio latente compartido entre voz y texto. La búsqueda de texto hacia audio es la dirección fuerte. De audio a texto se usa una cabeza CTC (cerca del 55 % de WER en ewé en las notas del repositorio). El inglés de LibriSpeech es la lengua de validación. Una demo de lista de reproducción puede asociar una consulta de texto con una pista.

El repositorio ya contiene `infer.py`, `eval.py`, `eval_ctc.py`, `app_playlist.py` y `run_playlist_app.sh`. Ese script de shell apunta a carpetas fijas del clúster del proyecto (`/home/data/projets-aps/projet6`). Usa la sección Usage de su `README.md` para el comando que corresponde a tu máquina. Instala antes los paquetes de `requirements.txt`.


La documentación en inglés que ya estaba en este repositorio sigue en el archivo en inglés.
