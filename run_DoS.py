from DoS.DoSFuncs import DoSFuncs

# use modified sample script from EXOSIMS
path = './sampleScript_coron.json'

# create DoSFuncs object with sample script and all default options
dos = DoSFuncs(path=path)
# plot depth of search
dos.plot_dos('all', 'Depth of Search')
# plot expected planets
dos.plot_nplan('all', 'Expected Planets')