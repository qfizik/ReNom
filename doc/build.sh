find rsts/api -type d -name autosum -print0 | xargs -0 rm -rf
rm -rf _build/html
if [ ! "$1" == 'NoJP' ]
  then
    make -e SPHINXOPTS="-D language='ja'" html
    mv _build/html _build/ja
fi

make html
mv _build/ja _build/html
