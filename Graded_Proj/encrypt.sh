#!/bin/bash

if [ ! -e slt2023_pub.pem ]
then
    echo "> Public key ./slt2023_pub.pem does not exist."
    exit 1
fi

notebookfile=$1

if [ ! -e "$notebookfile" ]
then
    echo "> Notebook file $notebookfile does not exist."
else
    openssl smime -encrypt -aes256 -in "$notebookfile" -binary -outform DEM \
        -out "$notebookfile.encr" slt2023_pub.pem
    
    echo "> File encrypted as $notebookfile.encr"
fi
