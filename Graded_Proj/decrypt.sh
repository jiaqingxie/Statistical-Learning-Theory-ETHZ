#!/bin/bash

if [ ! -e ~/.ssh/slt2023_prv.pem ]
then
    echo "> Private key ~/.ssh/slt2023_prv.pem does not exist."
    exit 1
fi

notebookfile=$1

if [ ! -e "$notebookfile" ]
then
    echo "> Notebook file $notebookfile does not exist."
else
    openssl smime -decrypt -aes256 -in "$notebookfile" -binary -inform DEM \
        -inkey ~/.ssh/slt2023_prv.pem -out "${notebookfile%'.encr'}"
    
    echo "> File decrypted as ${notebookfile%'.encr'}"
fi
