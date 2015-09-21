# This is a recipe for building the current development package into a conda binary.

The installation on travis-ci is done by building the conda package, installing
it, running the tests, and then if successful pushing the package to binstar
(and optionally pushing docs to AWS S3).

The binstar auth token is an encrypted environment variable generated using:

You set up travis to store an encrypted token via
```
gem install travis
travis encrypt BINSTAR_TOKEN=`binstar auth -n $PACKAGENAME -o $ORGNAME --max-age 22896000 -c --scopes api:write`
```
where `$PACKAGENAME` is the name of the package, and `$ORGNAME` the name of the binstar organization.

The final command should print a line (containing 'secure') for inclusion in your `.travis.yml` file.
