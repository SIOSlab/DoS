from DoS.DoSFuncs import DoSFuncs

"""
Calculates depth-of-search using the DoSFuncs object and the sample script sampleScript_coron.json
for EXOSIMS. Plots for depth-of-search and expected planets are produced.
"""
# use modified sample script from EXOSIMS
path = './sampleScript_coron.json'

# create DoSFuncs object with sample script and all default options
dos = DoSFuncs(path=path)
# plot depth of search
dos.plot_dos('all', 'Depth of Search')
# plot expected planets
dos.plot_nplan('all', 'Expected Planets')