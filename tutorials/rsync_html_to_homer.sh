
#scp index.html clawpack@homer.u.washington.edu:public_html/tidal-examples/

rsync -avz html_files/ \
    clawpack@homer.u.washington.edu:public_html/tidal-examples/tutorials/
