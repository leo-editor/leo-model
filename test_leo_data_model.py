#@+leo-ver=5-thin
#@+node:vitalije.20180510154246.1: * @file test_leo_data_model.py
from leoDataModel import (loadLeo, ltm_from_derived_file, p_to_lines,
                          auto_py, p_to_autolines, read_py_file_content)
import timeit
import hashlib
import os
import re
# import sys
### if len(sys.argv) < 2:
###     print('usage: python test_leo_data_model.py <leo instalation folder>')
###    sys.exit(1)
### LEOFOLDER = sys.argv[1]
### LEOCORE = os.path.join(LEOFOLDER, 'leo', 'core')
LEOCORE = 'c:/leo.repo/leo-editor/leo/core'
assert os.path.exists(LEOCORE)
#@+others
#@+node:vitalije.20180511114812.1: ** calcPaths
def calcPaths(ltm):
    res = {}
    pat = re.compile(r'^@path\s+(.+)$', re.M)
    npath = lambda x: os.path.normpath(os.path.abspath(x))
    cdir = os.path.abspath(LEOCORE)
    jpath = lambda x:npath(os.path.join(cdir, x))
    for gnx in ltm.nodes:
        [h, b] = ltm.attrs[gnx][:2]
        if h.startswith('@path '):
            pth = h[6:].strip()
            cdir = jpath(pth)
            continue
        m = pat.search(b)
        if m:
            cdir = jpath(m.group(1).strip())
            continue
        if h.startswith('@file '):
            pth = h[6:].strip()
            f = jpath(pth)
            res[gnx] = f
    return res
#@+node:vitalije.20180511095907.1: ** test_0
def test_0():
    global ltm, nloaded, pths
    nloaded = 0
    ### ltm = loadLeo(os.path.join(LEOCORE, 'LeoPyVitalije.leo'))
    ltm = loadLeo(os.path.join(LEOCORE, 'leoPy.leo'))
    pths = calcPaths(ltm)
    for i, pos in enumerate(ltm.positions):
        gnx = ltm.nodes[i]
        if gnx in pths:
            if ltm.attrs[gnx][0].startswith('@file '):
                ltm2 = ltm_from_derived_file(pths[gnx])
                ltm.replaceNode(ltm2)
                nloaded += 1
            elif ltm.attrs[gnx][0].startswith('@edit '):
                ltm.attrs[gnx][1] = open(pths[gnx], 'rt').read()
            else:
                continue
    return ltm
t1 = timeit.timeit(test_0, number=1)*1000
print('tree size', len(ltm.positions), 'read in %.2fms'%t1, 'files:%d'%nloaded)

#@+node:vitalije.20180511095921.1: ** m5
def m5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

#@+node:vitalije.20180514143806.1: ** dumpPos
def dumpPos(ltm, i):
    gnx = ltm.nodes[i]
    lev = ltm.levels[i]
    h, b, ps, cn, sz = ltm.attrs[gnx]
    if b is None:
        print(h, gnx)
    return repr((lev, gnx, h, m5(b.strip()), len(ps) > 1))

#@+node:vitalije.20180512101031.1: ** delims
def delims(pth):
    if pth.endswith('Attic.txt'):
        return '#', ''
    if pth.endswith('.txt'):
        return '.. ', ''
    if pth.endswith(('.py','.spec')):
        return '#', ''
    return '#unknown', ''

#@+node:vitalije.20180511095950.1: ** test_1
def test_1():
    ltm2 = ltm_from_derived_file(os.path.join(LEOCORE, 'leoGlobals.py'))
    return ltm2
ltm2 = test_1()
print('ok', len(ltm2.positions), ltm2.attrs[ltm2.nodes[0]][4])
t1 = timeit.timeit(test_1, number=5)/5*1000
print('Average: %.2fms'%t1)

#@+node:vitalije.20180511101400.1: ** test_tree_is_correct
def test_tree_is_correct(ltm):
    def reportNodes(gnx1, gnx2):
        i = ltm.nodes.index(gnx1)
        j = ltm.nodes.index(gnx2)
        return '\n'.join(('', dumpPos(ltm, i), dumpPos(ltm, j)))
    allgnx = set(ltm.nodes[1:])
    try:
        #@+others
        #@+node:vitalije.20180511101419.1: *3* parent-child links
        # check to see that every parent child link is  mutual
        for gnx in allgnx:
            for pgnx in ltm.parents(gnx):
                assert gnx in ltm.children(pgnx), 'FAIL parent/child 1' + repr((gnx, pgnx)) + '\n' + reportNodes(gnx, pgnx)
            for cgnx in ltm.children(gnx):
                assert gnx in ltm.parents(cgnx), 'FAIL parent/child 2' + reportNodes(gnx, cgnx)
        rgnx = ltm.nodes[0]
        for cgnx in ltm.children(rgnx):
            assert rgnx in ltm.parents(cgnx), 'FAIL parent/child 3' + reportNodes(cgnx, rgnx)

        #@+node:vitalije.20180512080054.1: *3* check size is correct
        for gnx in allgnx | set([ltm.nodes[0]]):
            sz = 1
            for cgnx in ltm.children(gnx):
                sz += ltm.attrs[cgnx][4]
            assert ltm.attrs[gnx][4] == sz, '%d != %r\n%r'%('FAIL SIZE', sz, ltm.attrs[gnx][4], reportNodes(gnx, gnx))

        #@+node:vitalije.20180512080620.1: *3* check parPos (parent positions)
        for i, gnx in enumerate(ltm.nodes):
            if i == 0:continue
            parp = ltm.parPos[i]
            pari = ltm.positions.index(parp)
            pgnx = ltm.nodes[pari]
            assert pgnx in ltm.parents(gnx), 'FAIL: parPos'

        #@+node:vitalije.20180512080632.1: *3* check gnx2pos
        for gnx, ps in ltm.gnx2pos.items():
            for p in ps:
                i = ltm.positions.index(p)
                assert ltm.nodes[i] == gnx, ('FAIL gnx2pos:', ltm.nodes[i], gnx)

        #@-others
    except Exception as e: 
        print('TEST FAILED')
        print(e)
#@+node:vitalije.20180512101005.1: ** test_writer
print('checking write')

def test_writer():
    for gnx, pth in pths.items():
        assert ltm.attrs[gnx][0].startswith('@file ')
        p1 = ltm.gnx2pos[gnx][0]
        lines = open(pth, 'rt').read().splitlines(True)
        a,b = delims(pth)
        for i, x in enumerate(p_to_lines(ltm, p1, a, b)):
            assert lines[i] == x, '\n'.join((
                ltm.attrs[gnx][0],
                'line:%d'%(i + 1),
                x[:-1],
                lines[i][:-1]))
test_writer()
print('ok')

#@+node:vitalije.20180529172206.1: ** test importer
def test_importer():
    print('test at-auto py import')
    table = (
        ('python 3', 'C:/Anaconda3/Lib'),
    ) 
    #@+others
    #@+node:ekr.20180530014828.1: *3* do_one_folder
    def do_one_folder(fold):
        todo = [fold]
        while todo:
            fold = todo.pop(0)
            for f in os.listdir(fold):
                fname = os.path.join(fold, f)
                ###
                # if fname in (
                    # '/home/vitalije/anaconda3/lib/python3.6/site-packages/pyflakes/test/test_api.py',
                    # ): continue
                if fname.endswith('test_api.py'):
                    continue
                if os.path.isdir(fname):
                    ### todo.append(fname)
                    continue
                if not f.endswith('.py'):
                    continue
                src = read_py_file_content(fname)
                tab = ' ' * 4
                ensurenl = lambda x:(x if x.endswith('\n') else x+'\n').replace('\t', tab)
                lines1 = [ensurenl(x) for x in src.splitlines(True) if x.strip()]
                if len(lines1) < 1: continue
                ltm = auto_py('rootgnx', fname)
                lines2 = [ensurenl(x[0]) for x in p_to_autolines(ltm, ltm.positions[0]) 
                                if x[0].strip()]
                if lines1 != lines2:
                    with open('/tmp/f1.py', 'w') as out:
                        out.write(''.join(lines1))
                    with open('/tmp/f2.py', 'w') as out:
                        out.write(''.join(lines2))
                assert lines1 == lines2, 'error importing file %s'%fname
    #@-others
    if 1:
        for name, path in table:
            print('importing: %s files' % name)
            do_one_folder(path)
    else:
        folders = ['/usr/lib/python2.7/', '/home/vitalije/anaconda3/lib/python3.6/']
        print('importing py2.7 files')
        do_one_folder(folders[0])
        print('importing py3.6 files')
        do_one_folder(folders[1])
    print('ok')
test_importer()
#@-others
ltm = test_0()
def tpickle():
    return ltm.tobytes()
ltmbytes = tpickle()

if 1: ###
    def unpickle():
        return ltm.restoreFromBytes(ltmbytes)
    
if 1:
    for x in ltm.nodes:
        if ltm.attrs[x][4] > 1 and len(ltm.gnx2pos[x]) > 1:
            p = ltm.gnx2pos[x][0]
            ltm.promote(p)
            test_tree_is_correct(ltm)
            break
    else:
        print('no clones found')
    test_tree_is_correct(ltm)
    print('tree correct  %d'%ltm.bytesSize())
    
if 0:
    t1 = timeit.timeit(tpickle, number=20)/20*1000
    print('pickle avg: %.2fms'%t1)
    t1 = timeit.timeit(unpickle, number=20)/20*1000
    print('upickle avg: %.2fms'%t1)
    print('profiling write_all')
    def test_write_all():
        for gnx, pth in pths.items():
            a, b = delims(pth)
            s = ltm.p_to_string(ltm.gnx2pos[gnx][0], a, b)
        assert s
    t1 = timeit.timeit(test_write_all, number=10)/10*1000
print('ok %.2fms'%t1)
#@-leo
