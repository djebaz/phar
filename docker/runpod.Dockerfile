FROM ghcr.io/djebaz/phar:torch1.8.0-cu111
# Force un shell interactif comme entrypoint
ENTRYPOINT ["/bin/bash","-lc"]
# Par défaut : rester vivant
CMD ["sleep infinity"]
