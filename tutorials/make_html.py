"""
Creates a subdirectory html_files containing rendered versions of each
notebook, along with .py and data files required to run them.  This
directory can be posted for people to browse the notebooks and modules
without needing to run them.

Some modifications are made to the notebooks before passing through
nbconvert, in particular replacing any references to other .ipynb 
notebooks by references to the .html version so that cross-referencing
between the notebooks works.

"""

import re
import subprocess
import os, sys
import glob


notebooks = ['Index','TidalForces','SpringNeapTides','TideOnSphere']

notebooks = ['SpringNeapTides']
# other files to copy to html_dir:
other_files = ['wake_island_tide.py','wake_island_tide_data.txt',
               'force_plots.py']
               
html_dir = './html_files'
os.system('mkdir -p %s' % html_dir)

for f in other_files:
    os.system('cp %s %s' % (f, html_dir))

for nb in notebooks:
    
    print("Processing %s, converting .ipynb to .html" % nb)
    
    input_filename = nb + '.ipynb'
    with open(input_filename, "r") as source:
        lines = source.readlines()
        
    os.chdir(html_dir)
    modified_filename = nb+'.ipynb'
    
    print('Modifying notebook and writing to %s' % html_dir)
    
    with open(modified_filename, "w") as output:
        for line in lines:
            #line = re.sub(r'from ipywidgets import interact', widget, line)
            #for nb_name in notebooks:
            #    line = re.sub(nb_name+'.ipynb', nb_name+'.html', line)
            line = re.sub('.ipynb', '.html', line)
            output.write(line)

    
    html_filename = os.path.join(nb+'.html')
    
    args = ["jupyter", "nbconvert", "--to", "html", "--execute",
            "--ExecutePreprocessor.kernel_name=python3",
            "--output", html_filename,
            "--ExecutePreprocessor.timeout=60", modified_filename]
            
            # "--template", template_path,
            
    subprocess.check_call(args)
    os.chdir('..')
