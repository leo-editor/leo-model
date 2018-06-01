import cProfile
import pstats
path = "c:/leo.repo/leo-editor/leo/core/leoPy.leo"
fn = 'profile_stats' # A binary file.
command = 'import test_leo_data_model'

# Doesn't work with multiprocessing: https://stackoverflow.com/questions/18150230
# command ='import miniTkLeo; miniTkLeo.main("%s")' % path

cProfile.run(command, fn)
p = pstats.Stats(fn).strip_dirs().sort_stats('tottime').print_stats('leoDataModel.py', 50)
