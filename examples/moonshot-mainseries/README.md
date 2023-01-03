# Relative free energy calculations of main medicinal chemistry milestone compounds for COVID Mooonshot

* Download and unpack Mpro source structures from Fragalysis using [permalink](https://fragalysis.diamond.ac.uk/viewer/react/download/tag/91448cc6-45c8-4707-94c3-3c59fc45c6da) into `structures/` subdirectory
* Generate a Spruce loop database from the retrieved structures
```bash
$OPENEYE_APLICATIONS/bin/loopdb_builder -in structures/aligned/ -source_name fragalysis -prefix mainseries
```


