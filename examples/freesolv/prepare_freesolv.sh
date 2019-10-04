#!/usr/bin/env bash
#Extract the SMILES from the database
#the first command excludes the first four lines
#the second command cuts by delimiter semicolon and takes the 3rd column
#the final command removes the leading whitespace in the file
tail -n +4 database.txt | cut -d ';' -f 2 | tr -d " " > database.smi

#use the openeye molgrep utility to find patterns matching the desired SMARTS
#string. here that is a string representing 6 membered aromatic rings
molgrep.py -p 'c1ccccc1' -i database.smi -o filtered_database.smi

#for convenience, depict all of the molecules that we've chosen:
mols2pdf.py -in filtered_database.smi -out filtered_database.pdf
mols2pdf.py -in database.smi -out full_database.pdf