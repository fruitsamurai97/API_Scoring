#!/bin/sh
# Navigue au répertoire où se trouve ton application Flask
#cd mon_projet
# Active ton environnement virtuel ici si nécessaire
# source /chemin/vers/monenv/bin/activate
# Lance Gunicorn avec les paramètres appropriés
gunicorn --bind=0.0.0.0 --timeout 60000 app:app
chmod +x startup.sh
  