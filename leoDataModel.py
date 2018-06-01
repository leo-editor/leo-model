#@+leo-ver=5-thin
#@+node:vitalije.20180510153405.1: * @file leoDataModel.py
import random
from collections import defaultdict, namedtuple
import re
import os
import xml.etree.ElementTree as ET
import pickle
import time
import leo.core.leoGlobals as g
assert g

class bunch:
    # pylint: disable=no-member
    def __init__(self, **kw):
        self.__dict__.update(kw)

#@+others
#@+node:vitalije.20180510153753.1: ** nthChild
def nthChild(ltm, pos, n):
    '''Returns index of nth child of a given position'''
    i = ltm.positions.index(pos)
    return nthChildI(ltm, i, n)

def nthChildI(ltm, i, n):
    '''Returns index of nth child of a given parent index'''
    gnx = ltm.nodes[i]
    h, b, ps, chn, sz = ltm.attrs[gnx]
    if len(chn) <= n: return -1
    i += 1
    while n:
        n -= 1
        cgnx = ltm.nodes[i]
        i += ltm.attrs[cgnx][4]
    return i

#@+node:vitalije.20180510153747.1: ** parPosIter
def parPosIter(ps, levs):
    '''Helper iterator for parent positions. Given sequence of
       positions and corresponding levels it generates sequence
       of parent positions'''
    rootpos = ps[0]
    levPars = [rootpos for i in range(256)] # max depth 255 levels
    it = zip(ps, levs)
    next(it) # skip first root node which has no parent
    yield rootpos
    for p, l in it:
        levPars[l] = p
        yield levPars[l - 1]

#@+node:vitalije.20180510153733.2: ** nodes2treemodel
def nodes2treemodel(nodes):
    '''
    Creates LeoTreeModel from the sequence of tuples
    (gnx, h, b, level, size, parentGnxes, childrenGnxes)
     0,   1, 2, 3,     4,    5,           6
    '''

    ltm = LeoTreeModel()
    ltm.positions = [random.random() for x in nodes]
    ltm.levels = [x[3] for x in nodes]
    ltm.parPos = list(parPosIter(ltm.positions, ltm.levels))
    ltm.nodes = [x[0] for x in nodes]
    gnx2pos = defaultdict(list)
    for pos, x in zip(ltm.positions, nodes):
        gnx2pos[x[0]].append(pos)
    ltm.gnx2pos = gnx2pos
    for gnx, h, b, lev, sz, ps, chn in nodes:
        ltm.attrs[gnx] = [h, b, ps, chn, sz[0]]
    # root node must not have parents
    rgnx = nodes[0][0]
    ltm.attrs[rgnx][2] = []
    return ltm

#@+node:vitalije.20180510153733.1: ** vnode2treemodel & viter
def vnode2treemodel(vnode):
    '''Utility convertor: converts VNode instance into
       LeoTreeModel instance'''
    def viter(v, lev0):
        s = [1]
        mnode = (v.gnx, v.h, v.b, lev0, s,
                [x.gnx for x in v.parents],
                [x.gnx for x in v.children])
        yield mnode
        for ch in v.children:
            for x in viter(ch, lev0 + 1):
                s[0] += 1
                yield x
    return nodes2treemodel(tuple(viter(vnode, 0)))
#@+node:vitalije.20180510153733.3: ** xml2treemodel
def xml2treemodel(xvroot, troot):
    '''Returns LeoTreeModel instance from vnodes and tnodes elements of xml Leo file'''
    parDict = defaultdict(list)
    hDict = {}
    bDict = dict((ch.attrib['tx'], ch.text or '') for ch in troot.getchildren())
    xDict = {}
    #@+others
    #@+node:vitalije.20180510153945.1: *3* xml_viter
    def xml_viter(xv, lev0, dumpingClone=False):
        s = [1]
        gnx = xv.attrib['t']
        if not xv: ### len(xv) == 0:
            # clone
            for ch in xml_viter(xDict[gnx], lev0, True):
                yield ch
            return
        chs = [ch.attrib['t'] for ch in xv if ch.tag == 'v']
        if not dumpingClone:
            xDict[gnx] = xv
            hDict[gnx] = xv[0].text
            for ch in chs:
                parDict[ch].append(gnx)
        mnode = [gnx, hDict[gnx], bDict.get(gnx, ''), lev0, s, parDict[gnx], chs]
        yield mnode
        for ch in xv.getchildren():
            if ch.tag != 'v':continue
            for x in xml_viter(ch, lev0 + 1, dumpingClone):
                s[0] += 1
                yield x

    #@+node:vitalije.20180510154050.1: *3* riter
    def riter():
        # EKR: Read iterator
        s = [1]
            # s: list of sizes
        chs = []
            # chs: list of child gnx's
        yield 'hidden-root-vnode-gnx', '<hidden root vnode>','', 0, s, [], chs
        for xv in xvroot.getchildren():
            gnx = xv.attrib['t']
            chs.append(gnx)
            parDict[gnx].append('hidden-root-vnode-gnx')
                # parDict: dict of parents of given gnx.
            for ch in xml_viter(xv, 1):
                s[0] += 1
                yield ch

    #@-others
    nodes = tuple(riter())
    return nodes2treemodel(nodes)

#@+node:vitalije.20180510153733.4: ** loadLeo
def loadLeo(fname):
    '''Loads given xml Leo file and returns LeoTreeModel instance'''
    # g.trace('=====',fname)
    with open(fname, 'rt') as inp:
        s = inp.read()
        xroot = ET.fromstring(s)
        vnodesEl = xroot.find('vnodes')
        tnodesEl = xroot.find('tnodes')
        ltm = xml2treemodel(vnodesEl, tnodesEl)
        return ltm
#@+node:vitalije.20180518104947.1: ** loadExternalFiles
def loadExternalFiles(ltm, loaddir):
    mpaths = paths(ltm, loaddir)
    for gnx, ps in mpaths.items():
        h = ltm.attrs[gnx][0]
        if h.startswith('@file '):
            ltm2 = ltm_from_derived_file(ps[0])
            ltm.replaceNode(ltm2)
        elif h.startswith('@auto ') and h.endswith('.py'):
            ltm2 = auto_py(gnx, ps[0])
            ltm2.attrs[gnx][0] = h
            ltm.replaceNode(ltm2)
    ltm.invalidate_visual()
#@+node:vitalije.20180518155338.1: ** loadLeoFull
def loadLeoFull(fname):
    '''Loads both given xml Leo file and external files.
       Returns LeoTreeModel instance'''
    ltm = loadLeo(fname)
    loaddir = os.path.dirname(fname)
    loaddir = os.path.normpath(loaddir)
    loaddir = os.path.abspath(loaddir)
    loadExternalFiles(ltm, loaddir)
    return ltm
#@+node:vitalije.20180518100350.1: ** paths
atFileNames = [
    "@auto-rst", "@auto","@asis",
    "@edit",
    "@file-asis", "@file-thin", "@file-nosent", "@file",
    "@clean", "@nosent",
    "@shadow",
    "@thin"
]
atFilePat = re.compile(r'^(%s)\s+(.+)$'%('|'.join(atFileNames)))

def paths(ltm, loaddir):
    '''Returns dict keys are gnx of each file node,
       and values are lists of absolute paths corresponding
       to the node.'''

    stack = [loaddir for x in range(255)]
    res = defaultdict(list)
    pat = re.compile(r'^@path\s+(.+)$', re.M)
    cdir = loaddir
    npath = lambda x: os.path.normpath(os.path.abspath(x))
    jpath = lambda x:npath(os.path.join(cdir, x))
    for p, gnx, lev in zip(ltm.positions, ltm.nodes, ltm.levels):
        if lev == 0: continue
        cdir = stack[lev - 1]
        h, b = ltm.attrs[gnx][:2]
        m = pat.search(h) or pat.search(b)
        if m:
            cdir = jpath(m.group(1))
            stack[lev] = cdir
        m = atFilePat.match(h)
        if m:
            res[gnx].append(jpath(m.group(2)))
    return res
#@+node:vitalije.20180510153738.1: ** class LeoTreeModel
class LeoTreeModel(object):
    '''Model representing all of Leo outline data.
       
       warning: work in progress - still doesn't contain all Leo data
       
       TODO: add support for unknownAttributes
             add support for status bits
             add support for gui view values
                    - body cursor position
                    - selected text'''

    def __init__(self):
        
        self.pickled_vars = (
            # for self.tobytes...
            'positions', 'nodes', 'attrs',
            'levels', 'gnx2pos', 'parPos',
            'expanded', 'marked', 'selectedPosition',
        )
        self.positions = []
        self.nodes = []
        self.attrs = {}
        self.levels = []
        self.parPos = []
        self.expanded = set()
        self.marked = set()
        self.selectedPosition = None
        self.gnx2pos = defaultdict(list)
        #
        self._visible_positions_serial = 0
        self._visible_positions_last = -1
        self._visible_positions = tuple()

    def ivars(self):
        # Note: toBytes pickles exactly these ivars.
        return (self.positions,
                self.nodes,
                self.attrs,
                self.levels,
                self.gnx2pos,
                self.parPos,
                self.expanded,
                self.marked,
                self.selectedPosition)

    #@+others
    #@+node:vitalije.20180510153738.2: *3* parents
    def parents(self, gnx):
        '''Returns list of gnxes of parents of node with given gnx'''
        a = self.attrs.get(gnx)
        return a[2] if a else []

    #@+node:vitalije.20180510153738.3: *3* children
    def children(self, gnx):
        '''Returns list of gnxes of children of the node with given gnx'''
        a = self.attrs.get(gnx)
        return a[3] if a else []

    #@+node:vitalije.20180516103839.1: *3* selectedIndex
    @property
    def selectedIndex(self):
        try:
            i = self.positions.index(self.selectedPosition)
        except ValueError:
            i = -1
        return i

    #@+node:vitalije.20180516160109.1: *3* insertLeaf (changed)
    def insertLeaf(self, pos, gnx, h, b):
        i = self.positions.index(pos)
        return self.insertLeafI(i, gnx, h, b)

    def insertLeafI(self, i, ignx, h, b):
        
        # ( positions, nodes, attrs, levels, gnx2pos, parPos,
          # expanded, marked, selPos) = self.ivars()
        attrs = self.attrs
        gnx2pos = self.gnx2pos
        levels = self.levels
        nodes = self.nodes
        parPos = self.parPos
        positions = self.positions
        # 
        pp = parPos[i]
        pi = positions.index(pp)
        pgnx = nodes[pi]
        lev = levels[pi] + 1
        levels[i:i] = [lev]
        positions[i:i] = [random.random()]
        parPos[i:i] = [pp]
        nodes[i:i] = [ignx]
        gnx2pos[ignx].append(positions[i])
        if ignx in attrs:
            attrs[ignx][2].append(pgnx)
        else:
            attrs[ignx] = [h, b, [pgnx], [], 1]
        
        def chiter(j):
            while levels[j] == lev:
                gnx = self.nodes[j]
                yield gnx
                j += attrs[gnx][4]
                if j >= len(levels): break

        attrs[pgnx][3] = list(chiter(pi + 1))
        
        def updateSize(x):
            attrs[x][4] += 1
            for px in attrs[x][2]:
                updateSize(px)

        updateSize(pgnx)
        return positions[i]
    #@+node:vitalije.20180510153738.4: *3* insertTree
    def insertTree(self, parent_gnx, index, t2):
        '''Inserts subtree in this outline as a child of node whose
           gnx is parent_gnx at given index'''
        gnx = t2.nodes[0]
        h, b = t2.attrs[gnx][:2]
        for pi in self.gnx2pos[parent_gnx]:
            i = nthChild(self, pi, index)
            self.insertLeafI(i, gnx, h, b)
        self.replaceNode(t2)
    #@+node:vitalije.20180510194736.1: *3* replaceNode & updateParentSize
    def replaceNode(self, t2):
        '''Replaces node with given subtree. This outline must contain
           node with the same gnx as root gnx of t2.'''
        t1 = self

        gnx = t2.nodes[0]
        sz0 = t1.attrs[gnx][4]

        # this function replaces one instance of given node
        def insOne(i):
            l0 = t1.levels[i]
            npos = [random.random() for x in t2.nodes]
            npos[0] = t1.positions[i]
            ppos = t1.parPos[i]
            t1.parPos[i:i+sz0] = list(parPosIter(npos, t2.levels))
            t1.parPos[i] = ppos
            t1.positions[i:i+sz0] = npos
            t1.nodes[i:i+sz0] = t2.nodes
            t1.levels[i:i+sz0] = [(l0 + x) for x in t2.levels]
            for p, gnx in zip(npos[1:], t2.nodes[1:]):
                t1.gnx2pos[gnx].append(p)
        # difference in sizes between old node and new node
        dsz = len(t2.positions) - sz0
        # parents of this node must be preserved
        t2.attrs[gnx][2] = t1.parents(gnx)
        for pi in t1.gnx2pos[gnx]:
            i = t1.positions.index(pi)
            insOne(i)
        # some of nodes in t2 may be clones of nodes in t1
        # they will have some parents that are outside t2
        # therefore it is necessary for these nodes in t2 to
        # update their parents list by adding only those parents
        # that are not part of t2.
        t2gnx = set(t2.nodes)
        for x in t2gnx:
            if x not in t1.attrs:continue
            if x is t2.nodes[0]:continue
            ps = t1.attrs[x][2]
            t2.attrs[x][2].extend([y for y in ps if y not in t2gnx])
        # now we can safely update attrs dict of t1
        t1.attrs.update(t2.attrs)
        # one last task is to update size in all ancestors of replaced node
        def updateParentSize(gnx):
            for pgnx in t1.parents(gnx):
                t1.attrs[pgnx][4] += dsz
                updateParentSize(pgnx)
        updateParentSize(gnx)
    #@+node:vitalije.20180510153738.5: *3* deleteNode (changed)
    def deleteNode(self, pos):
        g.trace('*****')
        # ( positions, nodes, attrs, levels, gnx2pos, parPos,
          # expanded, marked, selPos) = self.ivars()
        attrs = self.attrs
        gnx2pos = self.gnx2pos
        levels = self.levels
        nodes = self.nodes
        parPos = self.parPos
        positions = self.positions
        #
        i = positions.index(pos)
        sz = attrs[nodes[i]][4]

        # this function removes deleted positions
        # from list of positions for a single node
        # when this list becomes empty, it removes
        # also entry for this node in attrs dict
        def removeOne(i):
            gnx = nodes[i]
            xs = gnx2pos[gnx]
            xs.remove(positions[i])
            if not xs:
                gnx2pos.pop(gnx)

        # reduce size of parent and returns its index
        def updateParentSize(i):
            pp = parPos[i]
            j = positions.index(pp)
            attrs[nodes[j]][4] -= sz
            return j
        # remove all nodes in subtree
        for j in range(i, i + sz):
            removeOne(j)
        # now reduce sizes of all ancestors
        j = i
        while j > 0:
            j = updateParentSize(j)
        pp = parPos[i]
        pi = positions.index(pp)
        attrs[nodes[i]][2].remove(nodes[pi])
        N = attrs[nodes[pi]][4] + pi

        def chiter():
            j = pi + 1
            while j < N:
                gnx = nodes[j]
                yield gnx
                j += attrs[gnx][4]

        # finally delete all entries from data
        del nodes[i:i+sz]
        del positions[i:i+sz]
        del parPos[i:i+sz]
        del levels[i:i+sz]
        attrs[nodes[pi]][3] = list(chiter())
    #@+node:vitalije.20180515122209.1: *3* display_items
    def display_items(self, skip=0, count=None):
        '''
        A generator yielding tuples for visible, non-skipped items:
        
        (pos, gnx, h, levels[i], plusMinusIcon, iconVal, selInd == i)
        '''
        # g.trace('skip', skip, 'count', count)
        Npos = len(self.positions)
        if count is None:
            count = Npos
        ( positions, nodes, attrs, levels, gnx2pos, parPos,
          expanded, marked, selPos) = self.ivars()
        #
        selInd = self.selectedIndex
        i = 1
        while count > 0 and i < Npos:
            gnx = nodes[i]
            pos = positions[i]
            h, b, parents, hasChildren, treeSize = attrs[gnx]
            exp = (pos in expanded)
            if skip > 0:
                skip -= 1
            else:
                # There is one less line to be drawn.
                count -= 1
                # Compute the iconVal for the icon box.
                iconVal = 1 if b else 0
                iconVal += 2 if gnx in marked else 0
                iconVal += 4 if len(parents) > 1 else 0
                # Compute the +- icon if the node has children.
                if hasChildren:
                    plusMinusIcon = 'minus' if exp else 'plus'
                else:
                    plusMinusIcon = 'none'
                # Yield a tuple describing the line to be drawn.
                yield pos, gnx, h, levels[i], plusMinusIcon, iconVal, selInd == i
            if hasChildren and exp:
                i += 1
            else:
                i += treeSize
    #@+node:vitalije.20180515155021.1: *3* select_next_node
    def select_next_node(self, ev=None):
        i = self.selectedIndex
        if i < 0: i = 1
        if self.selectedPosition in self.expanded:
            i += 1
        else:
            i += self.attrs[self.nodes[i]][4]
        if i < len(self.positions):
            self.selectedPosition = self.positions[i]
            return self.nodes[i]

    #@+node:vitalije.20180515155026.1: *3* select_prev_node
    def select_prev_node(self, ev=None):
        i = self.selectedIndex
        if i < 0: i = 1
        j = j0 = 1
        while j < i:
            j0 = j
            if self.positions[j] in self.expanded:
                j += 1
            else:
                j += self.attrs[self.nodes[j]][4]
        self.selectedPosition = self.positions[j0]
        return self.nodes[j0]

    #@+node:vitalije.20180516103325.1: *3* select_node_left
    def select_node_left(self, ev=None):
        '''If currently selected node is collapsed or has no
           children selects parent node. If it is expanded and
           has children collapses selected node'''
        if self.selectedPosition in self.expanded:
            self.expanded.remove(self.selectedPosition)
            self.invalidate_visual()
            return
        i = self.selectedIndex
        if i < 2:return
        p = self.parPos[i]
        if p == self.positions[0]:
            # this is top level node
            # let's find previous top level (level=1) node.
            j = next(x for x in range(i - 1, 0, -1) if self.levels[x] == 1)
            p = self.positions[j]

        self.selectedPosition = p
        return self.nodes[self.positions.index(p)]
    #@+node:vitalije.20180516105003.1: *3* select_node_right
    def select_node_right(self, ev=None):
        '''If currently selected node is collapsed, expands it.
           In any case selects next node.'''
        i = self.selectedIndex
        if -1 < i < len(self.nodes) - 1:
            hasChildren = self.levels[i] < self.levels[i + 1]
            p = self.selectedPosition
            if hasChildren and p not in self.expanded:
                self.expanded.add(p)
                self.invalidate_visual()
            self.selectedPosition = self.positions[i + 1]
            return self.nodes[i + 1]
    #@+node:vitalije.20180518124047.1: *3* visible_positions
    @property
    def visible_positions(self):
        if self._visible_positions_serial != self._visible_positions_last:
            return self.refresh_visible_positions()
        return self._visible_positions

    def invalidate_visual(self):
        self._visible_positions_serial += 1

    def refresh_visible_positions(self):
        self._visible_positions_last = self._visible_positions_serial
        def refreshiter():
            attrs = self.attrs
            nodes = self.nodes
            positions = self.positions
            expanded = self.expanded
            j = 1
            N = len(positions)
            while j < N:
                p1 = positions[j]
                yield p1
                if p1 in expanded:
                    j += 1
                else:
                    j += attrs[nodes[j]][4]
        self._visible_positions = tuple(refreshiter())
        return self._visible_positions
    #@+node:vitalije.20180516150626.1: *3* subtree (changed)
    def subtree(self, pos):
        g.trace('*****')
        # ( positions, nodes, attrs, levels, gnx2pos, parPos,
          # expanded, marked, selPos) = self.ivars()
        attrs = self.attrs
        expanded = self.expanded
        levels = self.levels
        marked = self.marked
        nodes = self.nodes
        parPos = self.parPos
        positions = self.positions
        #
        i = positions.index(pos)
        gnx = nodes[i]
        sz = attrs[gnx][4]
        t = LeoTreeModel()
        t.positions = positions[i:i+sz]
        lev0 = levels[i]
        t.levels = [x - lev0 for x in levels[i:i+sz]]
        t.nodes = nodes[i:i+sz]
        t.parPos = parPos[i:i+sz]
        t.parPos[0] = 0
        for p, x in zip(t.positions, t.nodes):
            t.gnx2pos[x].append(p)
            if p in expanded:
                t.expanded.add(p)
            if p in marked:
                t.marked.add(p)

        knownGnx = set(t.nodes)
        for x in t.nodes:
            h, b, ps, chn, sz = attrs[x]
            ps = [y for y in ps if y in knownGnx]
            t.attrs[x] = [h, b, ps, chn[:], sz]
        return t

    #@+node:vitalije.20180516132431.1: *3* promote (changed)
    def promote(self, pos):
        '''Makes following siblings of pos, children of pos'''
        g.trace('*****')
        # ( positions, nodes, attrs, levels, gnx2pos, parPos,
          # expanded, marked, selPos) = self.ivars()
        attrs = self.attrs
        expanded = self.expanded
        gnx2pos = self.gnx2pos
        levels = self.levels
        nodes = self.nodes
        parPos = self.parPos
        positions = self.positions
        # 
        # node at position pos
        i = positions.index(pos)
        gnx = nodes[i]
        lev0 = levels[i]

        # parent node
        pp = parPos[i]
        pi = positions.index(pp)
        pgnx = nodes[pi]
        psz = attrs[pgnx][4]

        # index of node after tree
        after = pi + psz

        # remember originial size
        oldsize = attrs[gnx][4]
        A = i + oldsize

        # check danger clones
        if gnx in nodes[A:after]:
            print('warning node can not be in its own subtree')
            return

        # adjust size of this node
        attrs[gnx][4] = oldsize + after - A
        #@+others
        #@+node:vitalije.20180517160150.1: *4* 1. promote this part of outline
        j = A # iterator of following siblings
        while j < after:
            cgnx = nodes[j]
            parPos[j] = pos
            h, b, ps, chn, sz = attrs[cgnx]

            # let's replace pgnx with gnx in parents
            ps[ps.index(pgnx)] = gnx

            # append to children of this node
            attrs[gnx][3].append(cgnx)

            # remove from children of pgnx
            attrs[pgnx][3].remove(cgnx)

            # next sibling
            j += sz

        levels[A:after] = [x + 1 for x in levels[A:after]]
        #@+node:vitalije.20180517160237.1: *4* 2. update clones of this node in outline
        # now we have already made all following siblings 
        # children of this node (gnx at position pos)
        # we need to insert the same nodes to
        # outline after each clone in the outline
        allpos = [x for x in gnx2pos[gnx] if x != pos]
        if not allpos: return expanded.add(pos)

        # prepare data for insertion
        sibgnx = nodes[A:after]
        siblev = [x-lev0  for x in levels[A:after]]

        # for parPosIter we need level 0
        levs = [0] + siblev

        while allpos:
            ipos = allpos.pop()
            ii = positions.index(ipos)

            # old index of after tree
            jj = ii + oldsize

            # insert in nodes
            nodes[jj:jj] = sibgnx

            # insert levels adjusted by level of this clone
            lev1 = levels[ii]
            levels[jj:jj] = [x + lev1 for x in siblev]

            # we need new positions for inserted nodes
            npos = [random.random() for x in siblev]
            positions[jj:jj] = npos

            # insert parPos
            npos.insert(0, ipos)
            ppos = list(parPosIter(npos, levs))
            parPos[jj:jj] = ppos[1:]

            # update gnx2pos for each new position
            for p1,x in zip(npos[1:], sibgnx):
                gnx2pos[x].append(p1)
        #@+node:vitalije.20180517160615.1: *4* 3. update sizes in outline
        def updateSize(x):
            d = attrs[x]
            d[4] += after - A
            for x1 in d[2]:
                updateSize(x1)

        allparents = attrs[gnx][2][:]
        allparents.remove(pgnx) # for one of them we have already updated size
        for x in allparents:
            updateSize(x)
        #@-others

        # finally to show indented nodes
        expanded.add(pos)

    #@+node:vitalije.20180516141710.1: *3* promote_children (changed)
    def promote_children(self, pos):
        '''Turns children to siblings of pos'''
        g.trace('*****')
        # ( positions, nodes, attrs, levels, gnx2pos, parPos,
          # expanded, marked, selPos) = self.ivars()
        attrs = self.attrs
        gnx2pos = self.gnx2pos
        levels = self.levels
        nodes = self.nodes
        parPos = self.parPos
        positions = self.positions

        # this node
        i = positions.index(pos)
        gnx = nodes[i]

        # parent node
        pp = parPos[i]
        pi = positions.index(pp)
        pgnx = nodes[pi]
        pchn = attrs[pgnx][3]

        h, b, mps, chn, sz0 = attrs[gnx]
        # node after
        B = i + sz0

        #@+others
        #@+node:vitalije.20180517171517.1: *4* 1. reduce levels
        # reduce levels
        levels[i+1:B] = [x - 1 for x in levels[i + 1:B]]
        #@+node:vitalije.20180517171551.1: *4* 2. iterate over direct children
        j = i + 1
        while j < B:
            cgnx = nodes[j]
            h, b, ps, gchn, sz = attrs[cgnx]

            # replace parent with grandparent
            ps[ps.index(gnx)] = pgnx

            # remove from children of this node
            chn.remove(cgnx)

            # append to grandparent's children
            pchn.append(cgnx)

            # adjust parPos
            parPos[j] = pp

            j += sz # next child
        #@+node:vitalije.20180517171628.1: *4* 3. set size to 1
        attrs[gnx][4] = 1
        #@+node:vitalije.20180517171705.1: *4* 4. process standalone clones of this node
        if len(mps) == 1:
            # there is no standalone clones
            return

        def updateSize(x):
            d = attrs[x]
            d[4] = d[4] - sz0 + 1
            for x1 in d[2]:
                updateSize(x1)

        for px in gnx2pos[gnx]:
            i1 = positions.index(px)
            if px != pos:
                # this is another clone
                for j in range(i1 + 1, i1 + sz0):
                    cgnx = nodes[j]
                    gnx2pos[cgnx].remove(positions[j])
                pi1 = positions.index(parPos[i1])
                updateSize(nodes[pi1])
                del positions[i1+1:i1+sz0]
                del levels[i1+1:i1+sz0]
                del nodes[i1+1:i1+sz0]
                del parPos[i1+1:i1+sz0]
        #@-others
    #@+node:vitalije.20180517172602.1: *3* indent_node (changed)
    def indent_node(self, pos):
        '''Moves right node at position pos'''
        g.trace('*****')
        # ( positions, nodes, attrs, levels, gnx2pos, parPos,
          # expanded, marked, selPos) = self.ivars()
        attrs = self.attrs
        expanded = self.expanded
        gnx2pos = self.gnx2pos
        levels = self.levels
        nodes = self.nodes
        parPos = self.parPos
        positions = self.positions

        # this node
        i = positions.index(pos)
        if levels[i-1] == levels[i] - 1:
            # if there is no previous siblings node
            # can't be moved right
            return
        gnx = nodes[i]
        lev0 = levels[i]

        # parent node
        pp = parPos[i]
        pi = positions.index(pp)
        pgnx = nodes[pi]

        h, b, mps, chn, sz0 = attrs[gnx]
        # node after
        B = i + sz0


        j = A = pi + 1
        while j < i:
            A = j
            j += attrs[nodes[j]][4]
        # in A is index of new parent
        npgnx = nodes[A]
        if npgnx == gnx:
            print('Warning can not move node to its own subtree')
            return
        #@+others
        #@+node:vitalije.20180517172756.1: *4* 1. increase levels
        levels[i:B] = [x+1 for x in levels[i:B]]
        #@+node:vitalije.20180517172946.1: *4* 2. link to new parent

        mps[mps.index(pgnx)] = npgnx
        parPos[i] = positions[A]

        # remove from children of old parent
        attrs[pgnx][3].remove(gnx)

        # add to new parent's children
        attrs[npgnx][3].append(gnx)
        #@+node:vitalije.20180517173423.1: *4* 3. process clones of new parent
        #@+node:vitalije.20180517181545.1: *5* 3.1 update sizes
        oldSize = attrs[npgnx][4]

        def updateSize(x):
            d = attrs[x]
            d[4] += sz0
            for x1 in d[2]:
                updateSize(x1)

        attrs[npgnx][4] += sz0
        xx = attrs[npgnx][2][:]
        xx.remove(pgnx)
        for x in xx:
            updateSize(x)
        #@+node:vitalije.20180517181729.1: *5* 3.2 prepare data for insertion
        pxA = positions[A]

        levs = [x-lev0 for x in levels[i:B]]
        nnds = nodes[i:B]
        #@+node:vitalije.20180517181737.1: *5* 3.3 insert data
        for px in gnx2pos[npgnx]:
            if px == pxA: continue
            i1 = positions.index(px)

            jj = i1 + oldSize
            lev1 = levels[i1]
            levels[jj:jj] = [x + lev1 for x in levs]
            npos = [random.random() for x in levs]
            positions[jj:jj] = npos
            parPos[jj:jj] = list(parPosIter([px] + npos, [0] + levs))[1:]
            nodes[jj:jj] = nnds
            for p, x in zip(npos, nnds):
                gnx2pos[x].append(p)
        #@-others

        # finally let's make node visible
        expanded.add(positions[A])

    #@+node:vitalije.20180517183334.1: *3* dedent_node (changed)
    def dedent_node(self, pos):
        '''Moves node left'''
        g.trace('*****')
        # ( positions, nodes, attrs, levels, gnx2pos, parPos,
          # expanded, marked, selPos) = self.ivars()
        attrs = self.attrs
        gnx2pos = self.gnx2pos
        levels = self.levels
        nodes = self.nodes
        parPos = self.parPos
        positions = self.positions

        # this node
        i = positions.index(pos)
        if levels[i] == 1:
            # can't move left
            return
        gnx = nodes[i]

        # parent node
        pp = parPos[i]
        pi = positions.index(pp)
        pgnx = nodes[pi]
        psz = attrs[pgnx][4]

        h, b, mps, chn, sz0 = attrs[gnx]

        # grandparent node
        gpp = parPos[pi]
        gpi = positions.index(gpp)
        gpgnx = nodes[gpi]

        di0 = i - gpi
        di1 = di0 + sz0
        di2 = pi - gpi
        di3 = di2 + psz

        # replace parent with grandparent
        mps[mps.index(pgnx)] = gpgnx

        def chiter(a, b, skipIndex):
            while a < b:
                gnx = nodes[a]
                if a != skipIndex:
                    yield gnx
                a += attrs[gnx][4]
        attrs[pgnx][3] = list(chiter(pi + 1, pi + psz, i))
        attrs[pgnx][4] -= sz0
        def movedata(j, ar):
            ar[j+di0: j+di3] = ar[j+di1:j+di3] + ar[j+di0:j+di1]

        for px in gnx2pos[pgnx]:
            pxi = positions.index(px)
            gxi = pxi - di2
            if gxi >= 0 and nodes[gxi] == gpgnx:
                # just move data if necessary
                if di1 != di3:
                    movedata(gxi, positions)
                    movedata(gxi, parPos)
                    movedata(gxi, nodes)
                    movedata(gxi, levels)
                levels[gxi+di3-sz0:gxi+di3] = [x - 1 for x in levels[gxi+di3-sz0:gxi+di3]]
                parPos[gxi+di3-sz0] = positions[gxi]
            else:
                # delete moved node
                j = pxi + di0 - di2
                k = j + sz0
                for xi in range(j, k):
                    gnx2pos[nodes[xi]].remove(positions[xi])
                del positions[j:k]
                del levels[j:k]
                del nodes[j:k]
                del parPos[j:k]

        def updateSize(x):
            attrs[x][4] -= sz0
            for x1 in attrs[x][2]:
                updateSize(x1)

        xx = attrs[pgnx][2][:]
        xx.remove(gpgnx)
        for x in xx:
            updateSize(x)

        B = attrs[gpgnx][4] + gpi
        attrs[gpgnx][3] = list(chiter(gpi+1, B, 0))
    #@+node:vitalije.20180518062711.1: *3* prev_visible_index (changed)
    def prev_visible_index(self, pos):
        '''Assuming this node is visible, search for previous
           visible node.'''
        g.trace('*****')
        # ( positions, nodes, attrs, levels, gnx2pos, parPos,
          # expanded, marked, selPos) = self.ivars()

        # this node
        i = self.positions.index(pos)
        # parent node
        pp = self.parPos[i]
        pi = self.positions.index(pp)
        j = pi + 1
        A = pi
        while j < i:
            A = j
            if self.positions[j] in self.expanded:
                j += 1
            else:
                j += self.attrs[self.nodes[j]][4]
        return A
    #@+node:vitalije.20180518082938.1: *3* next_visible_index
    def next_visible_index(self, pos):
        '''Assuming this node is visible, search for previous
           visible node.'''

        # this node
        i = self.positions.index(pos)

        if pos in self.expanded:
            return i + 1
        return i + self.attrs[self.nodes[i]][4]
    #@+node:vitalije.20180518055719.1: *3* move_node_up
    def move_node_up(self, pos):
        '''Moves node one step towards the top of outline'''
        if pos == self.positions[1]:
            # already at the top
            return
        i = self.positions.index(pos)
        B = self.prev_visible_index(pos)

        return self.move_node_to_index(i, B)
    #@+node:vitalije.20180518055819.1: *3* move_node_down
    def move_node_down(self, pos):
        '''Moves node one step towards the end of outline'''
        positions = self.positions
        if pos == positions[-1]:
            # already at the end
            return
        nodes = self.nodes
        attrs = self.attrs
        expanded = self.expanded
        i = positions.index(pos)
        j = i + attrs[nodes[i]][4]
        if positions[j] in expanded:
            j += 1
        else:
            j += attrs[nodes[j]][4]

        return self.move_node_to_index(i, j)
    #@+node:vitalije.20180518071115.1: *3* move_node_to_index (changed)
    def move_node_to_index(self, A, B):
        '''Moves node from index A to index B'''
        g.trace('*****')
        # ( positions, nodes, attrs, levels, gnx2pos, parPos,
          # expanded, marked, selPos) = self.ivars()
        attrs = self.attrs
        gnx2pos = self.gnx2pos
        levels = self.levels
        nodes = self.nodes
        parPos = self.parPos
        positions = self.positions
        #
        gnx = nodes[A]
        sz0 = attrs[gnx][4]

        pp = parPos[A]
        pi = positions.index(pp)
        pgnx = nodes[pi]

        np = parPos[B]
        ndp = positions[B]
        npi = positions.index(np)
        npgnx = nodes[npi]

        def chiter(a, b, skip):
            while a < b:
                if a != skip:
                    yield nodes[a]
                a += attrs[nodes[a]][4]

        attrs[pgnx][3] = list(chiter(pi+1, pi+attrs[pgnx][4], A))
        ps = attrs[gnx][2]
        ps[ps.index(pgnx)] = npgnx

        def updateSize(x, d):
            attrs[x][4] += d
            for x1 in attrs[x][2]:
                updateSize(x1, d)

        updateSize(pgnx, -sz0)
        updateSize(npgnx, sz0)

        levs = [levels[x] - levels[A] + 1 for x in range(A, A + sz0)]
        nds = nodes[A:A+sz0]
        npos = positions[A:A+sz0]

        def movedata(a, b, c, arr):
            x = arr[a:c]
            if b < a:
                del arr[a:c]
                arr[b:b] = x
            else:
                arr[b:b] = x
                del arr[a:c]

        done = []
        for px in gnx2pos[pgnx]:
            pxi = positions.index(px)
            pb = positions[pxi-pi+B]
            a = pxi - pi + A
            c = a + sz0
            if pb in gnx2pos[npgnx]:
                done.append(pb)
                b = positions.index(pb)
                movedata(a, b, c, positions)
                movedata(a, b, c, parPos)
                movedata(a, b, c, nodes)
                li = levels[b]
                levels[a:c] = [li + x for x in levs]
                movedata(a, b, c, levels)
            else:
                for p, x in zip(positions[a:c], nodes[a:c]):
                    gnx2pos[x].remove(p)
                del positions[a:c]
                del nodes[a:c]
                del levels[a:c]
                del parPos[a:c]

        # we calculate this after data has been removed from outline
        db = positions.index(ndp) - positions.index(np)
        for px in gnx2pos[npgnx]:
            if px in done: continue
            pxi = positions.index(px)
            b = pxi + db
            if npos[0] in positions:
                npos = [random.random() for x in levs]
            for p, x in zip(npos, nds):
                gnx2pos[x].append(p)
            positions[b:b] = npos
            li = levels[pxi]
            levels[b:b] = [li + x for x in levs]
            nodes[b:b] = nds
            parPos[b:b] = list(parPosIter([px] + npos, [0] + levs))[1:]
        px = gnx2pos[npgnx][0]
        pxi = positions.index(px)
        attrs[npgnx][3] = list(chiter(pxi+1, pxi + attrs[npgnx][4], 0))
            # update attrs[npgnx][3], not attrs itself
        
    #@+node:vitalije.20180518155629.1: *3* body_change
    def body_change(self, newbody):
        i = self.selectedIndex
        if i < 0:
            return
        gnx = self.nodes[i]
        a = self.attrs[gnx]
        if a[1] != newbody:
            a[1] = newbody
    #@+node:vitalije.20180518155645.1: *3* p_to_string
    def p_to_string(self, p, delim_st, delim_en):
        '''Produces and returns string content of external file
           that corresponds to the given position, using provided
           delimiters.'''
        return ''.join(p_to_chunks(self, p, delim_st, delim_en))
    #@+node:vitalije.20180529164742.1: *3* p_to_autostring
    def p_to_autostring(self, p):
        '''Produces and returns string content of external auto file
           that corresponds to the given position.'''

        return ''.join(x[0] for x in p_to_autolines(self, p))

    #@+node:vitalije.20180518155650.1: *3* bytesSize
    def bytesSize(self):
        '''Test utility: returns length of pickled data of this model.'''
        return len(self.tobytes())
    #@+node:vitalije.20180518155655.1: *3* tobytes
    def tobytes(self):
        '''Returns pickled data of this model, for testing'''
        # data = self.ivars()
        data = [getattr(self, var) for var in self.pickled_vars]
        return pickle.dumps(data)
    #@+node:vitalije.20180518155712.1: *3* frombytes
    @staticmethod
    def frombytes(bs):
        ltm = LeoTreeModel()
        return ltm.restoreFromBytes(bs)

    @staticmethod
    def fromXml(fname):
        return loadLeo(fname)

    def loadExternalFiles(self, loaddir):
        return loadExternalFiles(self, loaddir)

    @staticmethod
    def loadFull(fname):
        return loadLeoFull(fname)
    #@+node:vitalije.20180518155716.1: *3* restoreFromBytes
    def restoreFromBytes(self, bs):
        data = pickle.loads(bs)
        ivarnames = ('positions,nodes,attrs,levels,gnx2pos,'
                     'parPos,expanded,marked,selectedPosition').split(',')
        for k, v in zip(ivarnames, data):
            setattr(self, k, v)
        self.invalidate_visual()
        return self
    #@-others


#@+node:vitalije.20180510103732.1: ** load_derived_file
def load_derived_file(lines):
    '''A generator yielding tuples: (gnx, h, b, level).'''
    # pylint: disable=no-member
    flines = tuple(enumerate(lines))
    #@+<< scan header, setting first_lines & delims >>
    #@+node:vitalije.20180510103732.2: *3* << scan header, setting first_lines & delims >>
    #@+at
    #    Find beginning of top (root node) of this derived file.
    #    We expect zero or more first lines before leo header line.
    # 
    #    Leo header line will give usefull information such as delimiters.
    #@@c
    header_pattern = re.compile(r'''
        ^(.+)@\+leo
        (-ver=(\d+))?
        (-thin)?
        (-encoding=(.*)(\.))?
        (.*)$''', re.VERBOSE)
    #
    # Scan for the header line, which follows any @first lines.
    first_lines = []
    for i, line in flines:
        m = header_pattern.match(line)
        if m:
            break
        first_lines.append(line)
    else:
        raise ValueError('wrong format, not derived file')
    #
    # Set the delims.
    # m.groups example ('#', '-ver=5', '5', '-thin', None, None, None, '')
    delim_st = m.group(1)
    delim_en = m.group(8)
    #@-<< scan header, setting first_lines & delims >>
    #@+<< define get_patterns >>
    #@+node:ekr.20180531102239.1: *3* << define get_patterns >>
    def get_patterns(delim_st, delim_en):
        '''
        Create regex patterns based on the given comment delims.
        There is zero cost to making this a function.
        '''
        if delim_en:
            dlms = re.escape(delim_st), re.escape(delim_en)
            # Plain...
            code_src = r'^%s@@c(ode)?%s$'%dlms
            doc_src = r'^%s@\+(at|doc)?(\s.*?)?%s$'%dlms
            ns_src = r'^(\s*)%s@\+node:([^:]+): \*(\d+)?(\*?) (.*?)%s$'%dlms
            sec_src = r'^(\s*)%s@(\+|-)<{2}[^>]+>>(.*?)%s$'%dlms
            # DOTALL..
            all_src = r'^(\s*)%s@(\+|-)all%s\s*$'%dlms
            oth_src = r'^(\s*)%s@(\+|-)others%s\s*$'%dlms
        else:
            dlms = re.escape(delim_st)
            all_src = r'^(\s*)%s@(\+|-)all\s*$'%dlms
            code_src = r'^%s@@c(ode)?$'%dlms
            doc_src = r'^%s@\+(at|doc)?(\s.*?)?'%dlms + '\n'
            ns_src = r'^(\s*)%s@\+node:([^:]+): \*(\d+)?(\*?) (.*)$'%dlms
            oth_src = r'^(\s*)%s@(\+|-)others\s*$'%dlms
            sec_src = r'^(\s*)%s@(\+|-)<{2}[^>]+>>(.*)$'%dlms
        ### union_src = '|'.join([code_src, doc_src, ns_src, sec_src])
        ### union2_src = '|'.join([all_src, oth_src])
        return bunch(
            all        = re.compile(all_src, re.DOTALL),
            code       = re.compile(code_src),
            doc        = re.compile(doc_src),
            node_start = re.compile(ns_src),
            others     = re.compile(oth_src, re.DOTALL),
            section    = re.compile(sec_src),
            ### union      = re.compile(union_src),
            ### union2     = re.compile(union2_src, re.DOTALL),
        )
    #@-<< define get_patterns >>
    patterns = get_patterns(delim_st, delim_en)
    #@+<< handle the top node >>
    #@+node:vitalije.20180510103732.5: *3* << handle the top node >>
    #@+<< define nodes bunch >>
    #@+node:vitalije.20180510103732.6: *4* << define nodes bunch >>
    # This bunch collects all input during the scan of input lines.
    nodes = bunch(
        level = defaultdict(list),
            # Keys are gnx's. Values are lists of node levels, in input order.
            # This is to support at-all directive which will write clones several times.
        head = {},
            # Keys are gnx's, values are the node's headline.
        body = defaultdict(list),
            # Keys are gnx's, values are list of body lines.
        gnxes = [],
            # A list of gnx, in the order in which they will be yielded.
    )
    #@-<< define nodes bunch >>
    #@+<< define set_node >>
    #@+node:vitalije.20180510103732.7: *4* << define set_node >>
    #@+at utility function to set data from regex match object from sentinel line
    #   see node_start pattern. groups[1 - 5] are:
    #   (indent, gnx, level-number, second star, headline)
    #       1      2         3            4          5
    #   returns gnx
    #@@c
    def set_node(m):
        gnx = m.group(2)
        lev = int(m.group(3)) if m.group(3) else 1 + len(m.group(4))
        nodes.level[gnx].append(lev)
        nodes.head[gnx] = m.group(5)
        nodes.gnxes.append(gnx)
        return gnx
    #@-<< define set_node >>
    #@+at
    # If the cloing comment delim, delim_en, exists
    # then doc parts start with delim_st alone on line
    # and doc parts end with delim_en alone on line
    # 
    # so whenever we encounter any of this lines, 
    # we have just to skip and nothing should be added to body.
    #@@c
    doc_skip = (delim_st + '\n', delim_en + '\n')
    #@+<< start top node >>
    #@+node:vitalije.20180510103732.9: *4* << start top node >>
    topnodeline = flines[len(first_lines) + 1][1] # line after header line
    m = patterns.node_start.match(topnodeline)
    topgnx = set_node(m)
    #
    # append first lines if we have some
    nodes.body[topgnx] = ['@first '+ x for x in first_lines]
    assert topgnx, 'top node line [%s] %d first lines'%(topnodeline, len(first_lines))
    #@-<< start top node >>
    #@-<< handle the top node >>
    #@+<< iterate all other lines >>
    #@+node:vitalije.20180510103732.10: *3* << iterate all other lines >>
    # Append lines to body, changing nodes as necessary.
    #@+<< init the grand iteration >>
    #@+node:ekr.20180531102949.1: *4* << init the grand iteration >>
    stack = []
        # Entries are (gnx, indent)
        # Updated when at+others, at+<section>, or at+all is seen.
    in_all = False
        # True: in @all.
    in_doc = False
        # True: in @doc parts.
    verbline = delim_st + '@verbatim' + delim_en + '\n'
        # The spelling of at-verbatim sentinel
    verbatim = False
        # True: the next line must be added without change.
    start = 2 * len(first_lines) + 2
        # Skip twice the number of first_lines, one header line, and one top node line
    indent = 0 
        # The current indentation.
    gnx = topgnx
        # The node that we are reading.
    body = nodes.body[gnx]
        # list of lines for current node.
    # dereference the patterns
    all_pat = patterns.all
    code_pat = patterns.code
    doc_pat = patterns.doc
    node_start_pat = patterns.node_start
    others_pat = patterns.others
    section_pat = patterns.section
    #@-<< init the grand iteration >>
    for i, line in flines[start:]:
        # These three sections must be first.
        #@+<< handle verbatim lines >>
        #@+node:vitalije.20180510103732.11: *4* << handle verbatim lines >>
        if verbatim:
            # Previous line was verbatim sentinel. Append this line as it is.
            body.append(line)
            verbatim = False # next line should be normally processed.
            continue
        if line == verbline:
            # This line is verbatim sentinel, next line should be appended as it is
            verbatim = True
            continue
        #@-<< handle verbatim lines >>
        #@+<< unindent the line if necessary >>
        #@+node:vitalije.20180510103732.12: *4* << unindent the line if necessary >>
        # is indent still valid?
        if indent and line[:indent].isspace() and len(line) > indent:
            # yes? let's strip unnecessary indentation
            line = line[indent:]
        #@-<< unindent the line if necessary >>
        #@+<< short-circuit later tests >>
        #@+node:ekr.20180531122608.1: *4* << short-circuit later tests >>
        # This is valid because all following sections are either:
        # 1. guarded by 'if in_doc' or
        # 2. guarded by a pattern that matches delim_st + '@'   
        if not in_doc and not line.strip().startswith(delim_st+'@'):
            body.append(line)
            continue
        #@-<< short-circuit later tests >>
        # The order of these sections does matter.
        #@+<< handle @all >>
        #@+node:vitalije.20180510103732.13: *4* << handle @all >>
        m = all_pat.match(line)
        if m:
            in_all = m.group(2) == '+' # is it opening or closing sentinel
            if in_all:
                # opening sentinel
                body.append('@all\n')
                # keep track which node should we continue to build
                # once we encounter closing at-all sentinel
                stack.append((gnx, indent))
            else:
                # this is closing sentinel
                # let's restore node where we started at-all directive
                gnx, indent = stack.pop()
                # restore body which should receive next lines
                body = nodes.body[gnx]
            continue
        #@-<< handle @all >>
        #@+<< handle @others >>
        #@+node:vitalije.20180510103732.14: *4* << handle @others >>
        m = others_pat.match(line)
        if m:
            in_doc = False
            if m.group(2) == '+': # is it opening or closing sentinel
                # opening sentinel
                body.append(m.group(1) + '@others\n')
                # keep track which node should we continue to build
                # once we encounter closing at-others sentinel
                stack.append((gnx, indent))
                indent += m.end(1) # adjust current identation
            else:
                # this is closing sentinel
                # let's restore node where we started at-others directive
                gnx, indent = stack.pop()
                # restore body which should receive next lines
                body = nodes.body[gnx]
            continue

        #@-<< handle @others >>
        #@+<< handle start of @doc parts >>
        #@+node:vitalije.20180510103732.15: *4* << handle start of @doc parts >>
        if not in_doc:
            # This guard ensures that the short-circuit tests are valid.
            m = doc_pat.match(line)
            if m:
                # yes we are at the beginning of doc part
                # was it @+at or @+doc?
                doc = '@doc' if m.group(1) == 'doc' else '@'
                doc2 = m.group(2) or '' # is there any text on first line?
                if doc2:
                    # start doc part with some text on the same line
                    body.append('%s%s\n'%(doc, doc2))
                else:
                    # no it is only directive on this line
                    body.append(doc + '\n')
                # following lines are part of doc block
                in_doc = True
                continue
        #@-<< handle start of @doc parts >>
        #@+<< handle start of @code parts >>
        #@+node:vitalije.20180510103732.16: *4* << handle start of @code parts >>
        if in_doc:
            # when using both delimiters, doc block starts with first delimiter
            # alone on line and at the end of doc block end delimiter is also
            # alone on line. Both of this lines should be skipped
            if line in doc_skip:
                continue
            #
            # maybe this line ends doc part and starts code part?
            m = code_pat.match(line)
            if m:
                # yes, this line is at-c or at-code line
                in_doc = False  # stop building doc part
                # append directive line
                body.append('@code\n' if m.group(1) else '@c\n')
                continue
        #@-<< handle start of @code parts >>
        #@+<< handle section refs >>
        #@+node:vitalije.20180510103732.17: *4* << handle section refs >>
        m = section_pat.match(line)
        if m:
            in_doc = False
            if m.group(2) == '+': # is it opening or closing sentinel
                # opening sentinel
                ii = m.end(2) # before <<
                jj = m.end(3) # at the end of line
                body.append(m.group(1) + line[ii:jj] + '\n')
                # keep track which node should we continue to build
                # once we encounter closing at-<< sentinel
                stack.append((gnx, indent))
                indent += m.end(1) # adjust current identation
            else:
                # this is closing sentinel
                # Restore node where we started at+<< directive
                gnx, indent = stack.pop()
                # Restore body which should receive next lines
                body = nodes.body[gnx]
            continue
        #@-<< handle section refs >>
        #@+<< handle node_start >>
        #@+node:vitalije.20180510103732.18: *4* << handle node_start >>
        m = node_start_pat.match(line)
        if m:
            in_doc = False
            gnx = set_node(m)
            if len(nodes.level[gnx]) > 1:
                # clone in at-all
                # let it collect lines in throwaway list
                body = []
            else:
                body = nodes.body[gnx]
            continue
        #@-<< handle node_start >>
        #@+<< handle @-leo >>
        #@+node:vitalije.20180510103732.19: *4* << handle @-leo >>
        if line.startswith(delim_st + '@-leo'):
            break
        #@-<< handle @-leo >>
        #@+<< handle directives >>
        #@+node:vitalije.20180510103732.20: *4* << handle directives >>
        if line.startswith(delim_st + '@@'):
            ii = len(delim_st) + 1 # on second '@'
            # strip delim_en if it is set or just '\n'
            jj = line.rfind(delim_en) if delim_en else -1
            # append directive line
            body.append(line[ii:jj] + '\n')
            continue
        #@-<< handle directives >>
        #@+<< handle in_doc >>
        #@+node:vitalije.20180510103732.21: *4* << handle in_doc >>
        if in_doc and not delim_en:
            # when using just one delimiter (start)
            # doc lines start with delimiter + ' '
            body.append(line[len(delim_st)+1:])
            continue
        #
        # when delim_en is not '', doc part starts with one start delimiter and \n
        # and ends with end delimiter followed by \n
        # in that case doc lines are unchcanged
        #@-<< handle in_doc >>
        #@afterref
 # Apparently, must be last.
        # A normal line.
        body.append(line)
    # Handle @last lines.
    if i + 1 < len(flines):
        nodes.body[topgnx].extend('@last %s'%x for x in flines[i+1:])
    #@-<< iterate all other lines >>
    #@+<< yield all nodes >>
    #@+node:vitalije.20180510103732.22: *3* << yield all nodes >>
    for gnx in nodes.gnxes:
        b = ''.join(nodes.body[gnx])
        h = nodes.head[gnx]
        lev = nodes.level[gnx].pop(0)
        yield gnx, h, b, lev-1
    #@-<< yield all nodes >>
#@+node:vitalije.20180510191554.1: ** ltm_from_derived_file & viter
def ltm_from_derived_file(fname):
    '''Reads external file and returns tree model.'''
    with open(fname, 'rt') as inp:
        lines = inp.read().splitlines(True)
        parents = defaultdict(list)
        #@+others
        #@+node:ekr.20180529183702.1: *3* viter (ltm_from_derived_file)
        def viter():
            stack = [None for i in range(256)]
            lev0 = 0
            for gnx, h, b, lev in load_derived_file(lines):
                ps = parents[gnx] # parents is a defaultdict(list)
                    # ps: parents
                cn = []
                    # cn: children
                s = [1]
                    # s: size, of some kind.
                stack[lev] = [gnx, h, b, lev, s, ps, cn]
                    #         0,   1, 2, 3,   4, 5,  6
                if lev:
                    # add parent gnx to list of parents
                    ps.append(stack[lev - 1][0])
                    if lev > lev0:
                        # parent level is lev0
                        # add this gnx to list of children in parent
                        stack[lev0][6].append(gnx)
                    else:
                        # parent level is one above 
                        # add this gnx to list of children in parent
                        stack[lev - 1][6].append(gnx)
                lev0 = lev
                # increase size of every node in current stack
                for z in stack[:lev]:
                    z[4][0] += 1
                # finally yield this node
                yield stack[lev]
        #@-others
        nodes = tuple(viter())
        return nodes2treemodel(nodes)
#@+node:vitalije.20180512100218.1: ** chunks2lines
def chunks2lines(it):
    '''Modifies iterator that yields arbitrary chunks of text
       to iterator that yields complete lines of text. This is
       used in testing for comparison with lines in external
       files.'''
    buf = []
    for line in it:
        if line.endswith('\n'):
            if buf:
                buf.append(line)
                yield ''.join(buf)
                buf = []
            else:
                yield line
        else:
            buf.append(line)
    if buf:
        yield ''.join(buf)

#@+node:vitalije.20180512100206.1: ** p_to_lines
def p_to_lines(ltm, pos, delim_st, delim_en):
    '''Returns iterator of lines representing the derived
       file (format version: 5-thin) of given position
       using provided delimiters.'''
    it = p_to_chunks(ltm, pos, delim_st, delim_en)
    return chunks2lines(it)
#@+node:vitalije.20180511214549.1: ** p_to_chunks
def p_to_chunks(ltm, pos, delim_st, delim_en):
    '''Returns iterator of chunks representing the derived
       file (format version: 5-thin) of given position
       using provided delimiters.'''

    for p, line, lln in p_at_file_iterator(ltm, pos, delim_st, delim_en):
        yield line
#@+node:vitalije.20180529171633.1: *3* p_at_file_iterator
def p_at_file_iterator(ltm, pos, delim_st, delim_en):
    last = []
    pindex = ltm.positions.index(pos)
    # pylint: disable=no-member
        # bunch confuses pylint.
    #@+others
    #@+node:vitalije.20180511214549.2: *4* conf
    conf = bunch(
        delim_st=delim_st,
        delim_en=delim_en,
        in_doc=False,
        in_all=False,
        zero_level=ltm.levels[pindex])
    #@+node:vitalije.20180511214549.3: *4* write patterns
    section_pat = re.compile(r'^(\s*)(<{2}[^>]+>>)(.*)$')

    others_pat = re.compile(r'^(\s*)@others\b', re.M)
        # !important re.M used also in others_iterator

    doc_pattern = re.compile('^(@doc|@)(?:\\s(.*?)\n|\n)$')

    code_pattern = re.compile('^(@code|@c)$')


    # TODO: check if there are more directives that
    #       should be in this pattern
    atdir_pat = re.compile('^@('
        'beautify|'
        'color|'
        'encoding|'
        'killbeautify|'
        'killcolor|'
        'language|'
        'last|'
        'nobeautify|'
        'nocolor-node|'
        'nocolor|'
        'nosearch|'
        'pagewidth|'
        'path|'
        'root|'
        'tabwidth|'
        'wrap)')
    #@+node:vitalije.20180511214549.4: *4* section_ref
    def section_ref(s):
        m = section_pat.match(s)
        if m:
            return m.groups()
        return None, None, None

    #@+node:vitalije.20180512071638.1: *4* shouldBeIgnored
    def shouldBeIgnored(h, b):
        return h.startswith('@ignore') or b.startswith('@ignore') or '\n@ignore' in b

    #@+node:vitalije.20180511214549.5: *4* others_iterator
    def others_iterator(pind):
        # index of node after subtree is pind + <subtree-size>
        after = pind + ltm.attrs[ltm.nodes[pind]][4]

        p1 = pind + 1
        while p1 != after:
            gnx = ltm.nodes[p1]
            h, b, ps, chn, sz = ltm.attrs[gnx]
            if shouldBeIgnored(h, b) or section_ref(h)[1]:
                p1 += sz  # skip entire p1-subtree 
            else:
                yield p1
                if others_pat.search(b):
                    # skip entire subtree if p1 contains at-others directive
                    p1 += sz
                else:
                    # next node in outline order
                    p1 += 1
    #@+node:vitalije.20180512160226.1: *4* findReference
    def findReference(pind, ref):
        '''Returns index of node with section definition.'''
        gnx = ltm.nodes[pind]
        sz = ltm.attrs[gnx][4]

        # clean and normalize reference
        ref = ref.lower().replace(' ', '').replace('\t', '')

        for i in range(pind+1, pind+sz): # reference must be in subtree
            gnx = ltm.nodes[i]
            h, b, ps, chn, sz = ltm.attrs[gnx]
            h = h.lower().replace(' ', '').replace('\t', '').lstrip('.')
            # normalize headline
            if h.startswith(ref):
                # found reference definition node
                return i

    #@+node:vitalije.20180511214549.6: *4* section_replacer
    def section_replacer(it):
        for p, w, final,lln in it:
            if final:
                # pass final lines
                yield p, w, final, lln
                continue
            # does this line contain section reference
            indent, sref, after = section_ref(w)
            if sref and not conf.in_doc:
                if conf.in_all:
                    #CHECK: This perhaps can't happen
                    yield p, w, True, lln
                else:
                    p1 = findReference(p, sref)
                    if not p1:
                        raise LookupError('unresolved section reference: %s'%w)
                    yield p, sent_line('@+', sref, indent=indent), True, lln
                    for p2, w2, final,lln2 in all_lines(p1):
                        w2 = indent + w2 if w2 != '\n' else w2
                        yield p2, w2, final, lln2
                    yield p, sent_line('@-', sref, after, indent=indent), True,lln
                    conf.in_doc = False
            else:
                yield p, w, final, lln
    #@+node:vitalije.20180511214549.7: *4* open_node
    def open_node(p):
        gnx = ltm.nodes[p]
        lev = ltm.levels[p]
        h = ltm.attrs[gnx][0]
        stlev = star_level(lev - conf.zero_level)
        return  sent_line('@+node:', gnx, stlev, h)

    #@+node:vitalije.20180511214549.8: *4* body_lines
    def body_lines(p):
        plevel = ltm.levels[p]
        first = plevel == conf.zero_level
        if not first:
            yield p, open_node(p), True, 0
        conf.in_doc = False
        b = ltm.attrs[ltm.nodes[p]][1]
        blines = b.splitlines(True)
        if b:
            for i, line in enumerate(blines):
                # child nodes should use continue
                # if they need to skip following nodes
                #@+others
                #@+node:vitalije.20180511214549.9: *5* verbatim
                verbatim = needs_verbatim(line)
                if verbatim:
                    yield p, sent_line('@verbatim'), True, i + 1
                    yield p, line, True, i + 1
                    continue

                #@+node:vitalije.20180511214549.10: *5* first lines & leo header
                if first:
                    if line.startswith('@first '):
                        yield p, line[7:], True, i + 1
                        continue
                    else:
                        first = False
                        yield p, sent_line('@+leo-ver=5-thin'), True, i + 1
                        yield p, open_node(p), True, i + 1
                        fstr = sent_line('@@first')
                        for k in range(i):
                            yield p, fstr, True, i + 1

                #@+node:vitalije.20180511214549.11: *5* last lines
                if not conf.in_all:
                    if line.startswith('@last '):
                        last.append((line[6:], i + 1))
                        yield p, sent_line('@@last'), True, i + 1
                        continue
                    elif last:
                        raise ValueError('@last must be last line in body')
                        ### break

                #@-others
                yield p, line, False, i + 1
            # pylint: disable=undefined-loop-variable
            if not line.endswith('\n'):
                yield p, '\n', False, len(blines)
    #@+node:vitalije.20180511214549.12: *4* needs_verbatim
    def needs_verbatim(line):
        return line.lstrip().startswith(conf.delim_st + '@')

    #@+node:vitalije.20180511214549.13: *4* others_replacer
    def others_replacer(it):
        for p, w, final, lln in it:
            if final:
                yield p, w, final, lln
                continue
            m = others_pat.match(w)
            if m and not conf.in_doc:
                if conf.in_all:
                    #CHECK: This perhaps can't happen
                    yield p, w, True, lln
                else:
                    indent = m.group(1)
                    w1 = sent_line('@+others',indent=indent)
                    yield p, w1, True, lln
                    for p1 in others_iterator(p):
                        for p2, w2, final, lln2 in all_lines(p1):
                            w2 = indent + w2 if w2 != '\n' else w2
                            yield p2, w2, final, lln2
                    yield p, sent_line('@-others',indent=indent), True, lln
                    conf.in_doc = False
            else:
                yield p, w, final, lln

    #@+node:vitalije.20180511214549.14: *4* atall_replacer
    def atall_replacer(it):
        for p, w, final, lln in it:
            if final:
                yield p, w, final, lln
                continue
            if w == '@all\n':
                conf.in_all = True
                conf.in_doc = False
                yield p, sent_line('@+all'), True, lln
                gnx = ltm.nodes[p]
                sz = ltm.attrs[gnx][4]
                for p1 in range(p + 1, p + sz):
                    for p2, w2, final, lln2 in all_lines(p1):
                        yield p2, w2, final, lln2
                yield p, sent_line('@-all'), True, lln
                conf.in_all = False
                conf.in_doc = False
            else:
                yield p, w, final, lln

    #@+node:vitalije.20180511214549.16: *4* all_lines
    def all_lines(p):
        it = body_lines(p)
        it = atall_replacer(it)
        it = section_replacer(it)
        it = others_replacer(it)
        it = at_adder(it)
        it = at_docer(it)
        return it
    #@+node:vitalije.20180511214549.17: *4* at_docer
    def at_docer(it):
        for p, w, final, lln in it:
            if final or conf.in_all:
                yield p, w, final, lln
                continue
            #@+others
            #@+node:vitalije.20180511214549.18: *5* at, at-doc
            if not conf.in_doc:
                m = doc_pattern.match(w)
                if m:
                    conf.in_doc = True
                    docdir = '@+at' if m.group(1) == '@' else '@+doc'
                    docline = ' ' + m.group(2) if m.group(2) else ''
                    yield p, sent_line(docdir, docline), True, lln
                    if conf.delim_en:
                        yield p, conf.delim_st + '\n', True, lln
                    continue

            #@+node:vitalije.20180511214549.19: *5* at-c at-code
            if conf.in_doc:
                m = code_pattern.match(w)
                if m:
                    if conf.delim_en:
                        yield p, conf.delim_en  + '\n', True, lln
                    yield p, sent_line('@', m.group(1)), True, lln
                    conf.in_doc = False
                    continue

            #@-others
            if conf.in_doc and not conf.delim_en:
                yield p, sent_line(' ', w[:-1]), True, lln
            else:
                yield p, w, False, lln

    #@+node:vitalije.20180511214549.20: *4* at_adder
    def at_adder(it):
        for p, w, final, lln in it:
            if final:
                yield p, w, final, lln
                continue
            m = atdir_pat.match(w)
            if m and not conf.in_all:
                yield p, sent_line('@', w[:-1]), True, lln
            else:
                yield p, w, False, lln

    #@+node:vitalije.20180511214549.21: *4* star_level
    def star_level(lev):
        # pylint: disable=consider-using-ternary
        if lev < 2:
            return [': * ', ': ** '][lev]
        else:
            return ': *%d* '%(lev + 1)

    #@+node:vitalije.20180511214549.22: *4* sent_line
    def sent_line(s1, s2='', s3='', s4='', indent=''):
        return ''.join((indent,conf.delim_st, s1, s2, s3, s4, conf.delim_en, '\n'))

    #@-others
    # pylint: disable=undefined-loop-variable
        # all_lines is not empty.
    for p, line, final, lln in all_lines(pindex):
        yield p, line, lln
    yield p, sent_line('@-leo'), 0
    for line, lln in last:
        yield p, line, lln
#@+node:vitalije.20180529143138.1: ** new_gnx
def new_gnx_generator(_id):
    nind = lambda:_id + time.strftime('.%Y%m%d%H%M%S', time.localtime())
    curr = [nind()]
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    def wrapper():
        ts = nind()
        if ts != curr[0]:
            curr[0] = ts
            return ts
        return ts + '.' + ''.join(random.choice(chars) for i in range(6))
    return wrapper
new_gnx = new_gnx_generator('vitalije')
#@+node:vitalije.20180529134759.1: ** auto_py
def auto_py(gnx, fname):
    '''Builds outline from py file'''
    return auto_py_from_string(gnx, read_py_file_content(fname))

def read_py_file_content(fname):
    # pylint: disable=len-as-condition
    pat = re.compile(br'^[ \t\v]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)')
    with open(fname, 'rb') as inp:
        bsrc = inp.read()
        lines = bsrc.split(b'\n', 2)
        l1 = lines[0] if len(lines) > 0 else b''
        l2 = lines[1] if len(lines) > 1 else b''
        m = pat.match(l1) or pat.match(l2)
        if m:
            enc = m.group(1).decode('ascii')
            if enc == 'iso-latin-1-unix':
                enc = 'latin-1'
        else:
            enc = 'utf-8'
        src = bsrc.decode(enc)
        if '\t' in src:
            src = src.replace('\t', ' '*4)
        return src
#@+node:vitalije.20180529135809.1: ** auto_py_from_string (changed 2)
Line = namedtuple('Line', 'ln st en ind cl isInd cpos txt')
def auto_py_from_string(rgnx, src):
    # init_import
    ind = 0
    attrs = {}
    attrs[rgnx] = [[], [], '', '']
    #@+others
    #@+node:vitalije.20180529152034.1: *3* add_lines
    def add_lines(slines, blines, ind):
        efind = ind
        for x in slines:
            if x.ind != 0 or x.cpos != 1:
                if x.ind < efind:
                    efind = x.ind
                    blines.append('@setindent %d\n'%x.ind)
                elif x.ind >= ind and efind != ind:
                    efind = ind
                    blines.append('@setindent %d\n'%ind)
            blines.append(x.txt[efind:])
        if efind != ind:
            blines.append('@setindent %d\n'%ind)
    #@+node:vitalije.20180529150815.1: *3* add_new_node
    def add_new_node(vr):
        gnx = new_gnx()
        attrs[vr][1].append(gnx)
        attrs[gnx] = [[vr], [], '', '']
        return gnx
    #@+node:vitalije.20180529151433.1: *3* first_in_block
    def first_in_block(x, slines):
        y = slines[x.ln + 1]
        while not y.cl or y.cpos - y.ind <= 1:
            y = slines[y.ln + 1]
        return y
    #@+node:vitalije.20180529152008.1: *3* hname
    def hname(x):
        if x.txt.startswith('def ', x.ind):
            return x.txt[x.ind+4:].split('(',1)[0]
        elif x.txt.startswith('class ', x.ind):
            s = x.txt[x.ind+6:] + '('
            i = min(s.find('('), s.find(':'))
            return s[:i]
        else:
            return x.txt[x.ind:]
    #@+node:vitalije.20180529151407.1: *3* inside_block
    def inside_block(x, slines):
        y = next_start(x, slines)
        y = first_in_block(y, slines)
        z = next_statement(y, x.ind, slines)
        return y, z.ln - x.ln
    #@+node:vitalije.20180529151523.1: *3* mkvnode
    def mkvnode(vr, x, sz, tl, slines):
        v = new_gnx()
        attrs[v] = [[vr], [], hname(x), '']
        attrs[vr][1].append(v)
        i = x.ln + sz
        N = slines[-1].ln
        f = lambda y:y.cl and y.cpos - y.ind <= 1
        f1 = lambda y:y.ln < N and not(y.txt.strip())
        while f(slines[i-1]):i -= 1
        while f1(slines[i]):i += 1
        return i, v
    #@+node:vitalije.20180529151422.1: *3* next_start
    def next_start(x, slines):
        while not x.isInd:
            x = slines[x.ln + 1]
        return x
    #@+node:vitalije.20180529151449.1: *3* next_statement
    def next_statement(x, lev, slines):
        y = slines[x.ln + 1]
        N = slines[-1].ln
        while y.ln < N:
            while not y.cl or y.cpos - y.ind <= 1:
                if y.ln < N:
                    y = slines[y.ln + 1]
                else:
                    break
            if y.ind <= lev:
                break
            elif y.ln < N:
                y = slines[y.ln + 1]
        return y
    #@+node:vitalije.20180529151243.1: *3* nodes_from_lines
    def nodes_from_lines(vr, ind, slines, a, b):
        tl = 0
        res = {}
        filt = lambda x: (
            x.cl and
            x.txt.startswith(('def ', 'class '), ind) and
            x.cpos - x.ind > 1)
        filt2 = lambda x: ':' in x.txt and x.txt.rsplit(':', 1)[-1].strip()
        fclines = [x for x in slines[a:b] if filt(x)]
        if not fclines:
            res[vr] = [None, tl, b, b, b, ind, None]
            return res
        for x in fclines:
            if filt2(x):
                fl, sz = x, 1
            else:
                fl, sz = inside_block(x, slines)
            if vr not in res:
                res[vr] = [None, tl, x.ln, x.ln+sz, b, ind, None]
                tl = x.ln
            tl2, v = mkvnode(vr, x, sz, tl, slines)
            res[v] = [x, tl, tl2, tl2, tl2, ind, fl]
            tl = tl2
        if tl < len(slines):
            res[vr][3] = tl
        return res
    #@+node:vitalije.20180529151834.1: *3* srcLines
    def srcLines(src):
        lines = src.splitlines(True)
        if lines[-1] != '\n':lines.append('\n')
        N = len(src)
        pos = [0]
        #@+others
        #@+node:vitalije.20180529151834.2: *4* find
        ### N = len(src)
        def find(chs, i):
            ii = [src.find(x, i) for x in chs]
            ii.append(N)
            return min([x for x in ii if x > -1])
        #@+node:vitalije.20180529151834.4: *4* isq3
        def isq3(i, x):
            while q3blocks[0][1] < i:
                q3blocks.pop(0)
            a, b = q3blocks[0]
            return a <= i <= b
        #@+node:vitalije.20180529151834.5: *4* isindent
        def isindent(i, x, cmi):
            a, b = q3blocks[0]
            return (a > i or b < cmi) and \
                x[:cmi - i].rstrip().endswith((':', ':\n'))
        #@+node:vitalije.20180529151834.6: *4* mkoneline

        def mkoneline(i, x):
            a = pos[0]
            cl = not isq3(a, x)
            while comments[0] < a:
                comments.pop(0)
            pos[0] = a + len(x)
            cmi = min(comments[0], pos[0])
            isInd = isindent(a, x, cmi)
            cpos = cmi - a
            xlstr = x.lstrip()
            ind = len(x) - len(xlstr) if xlstr else 0
            if not xlstr:
                x = '\n'
                cpos = 1
            return Line(i, a, pos[0], ind, cl, isInd, cpos, x)
        #@-others
        #@+<< q3blocks and comments >>
        #@+node:vitalije.20180529151834.3: *4* << q3blocks and comments >>
        q3s = "'''"
        q3d = '"""'
        q3sd = (q3s, q3d)
        q3blocks = []
        comments = []
        i = 0
        while i < N:
            i1 = find(("'", '"', '#'), i)
            if i1 == N:break
            if src.startswith(q3sd, i1):
                i2 = find(q3sd, i1 + 3)
                q3blocks.append((i1, i2))
                i = i2 + 3
            elif src[i1] in ('"', "'"):
                i2 = i1 + 1
                while src[i2] != src[i1]:
                    if src[i2] == '\\':
                        i2 += 2
                    elif src[i2] == '\n':
                        break
                    else:
                        i2 += 1
                q3blocks.append((i1, i2))
                i = i2 + 1
            elif src[i1] == '#':
                i = find(('\n',),  i1 + 1)  + 1
                comments.append(i1)
            else:
                i = i1
        q3blocks.append((N,N))
        comments.append(N)
        #@-<< q3blocks and comments >>
        return [mkoneline(i, x) for i, x in enumerate(lines)]
    slines = srcLines(src)
    #@+node:ekr.20180530114405.1: *3* viter
    def viter(gnx, lev0):
        s = [1]
        ps, chn, h, b = attrs[gnx]
        mnode = (gnx, h, b, lev0, s, ps, chn)
        yield mnode
        for ch in chn:
            for x in viter(ch, lev0 + 1):
                s[0] += 1
                yield x
    #@-others
    #@+<< do import >>
    #@+node:vitalije.20180529151122.1: *3* << do import >>
    res = nodes_from_lines(rgnx, ind, slines, 0, len(slines))
    todo = set(res.keys())
    while todo:
        gnx = todo.pop()
        x, a, b, c, d, ind, fl = res[gnx]
        if b - a > 30 and x and x.txt.startswith('class ', x.ind):
            r1 = nodes_from_lines(gnx, fl.ind, slines, fl.ln, b)
            b, c, d = r1.pop(gnx)[2:5]
            res[gnx] = x, a, b, c, d, ind, fl
            res.update(r1)
            todo.update(r1.keys())
    #@-<< do import >>
    #@+<< fill lines >>
    #@+node:vitalije.20180529151151.1: *3* << fill lines >>
    for gnx in res:
        x, a, b, c, d, ind, fl = res[gnx]
        atr = attrs[gnx]
        if x:
            atr[2] = hname(x)
        blines = []
        if fl:
            add_lines(slines[a:fl.ln], blines, ind)
            add_lines(slines[fl.ln:b], blines, ind)
            if d >= c > b:
                blines.append((' '* (fl.ind - ind)) + '@others\n')
                add_lines(slines[c:d], blines, ind)
        else:
            add_lines(slines[a:b], blines, 0)
            if d >= c > b:
                blines.append('@others\n')
                add_lines(slines[c:d], blines, 0)

        atr[3] = ''.join(blines)
    #@-<< fill lines >>
    return nodes2treemodel(tuple(viter(rgnx, 0)))
#@+node:vitalije.20180529163709.1: ** p_to_autolines (changed 2)
def p_to_autolines(ltm, pos):
    '''Returns an iterator of lines generated by at-auto node p.
       It respects at-others and at-setindent directives. Does
       not expand section references and treats section nodes
       as ordinary ones, puts their content among other nodes.
       
       yields tuples of (line, ni, lln) where 
            line is text content
            ni - index of currently outputing node
            lln - local line number, i.e. line number in the
                  currently outputing body
    '''
    NI = [ltm.positions.index(pos)]
    #@+others
    #@+node:vitalije.20180529163709.2: *3* all_lines (changed 2)
    def all_lines(gnx, ind):
        mNI = NI[0]
        h, b, ps, chn, sz = ltm.attrs[gnx]
        lines = b.splitlines(True)
        for i, line in enumerate(lines):
            ### Removed implicit section reference.
            if line.startswith('@setindent '):
                ind = int(line[11:].strip())
                continue
            if line.startswith(('@killcolor\n', '@nocolor\n','@language ')):
                continue
            if '@others' in line:
                sline = line.lstrip()
                ws = len(line) - len(sline)
                if sline == '@others\n':
                    for ch in chn:
                        NI[0] += 1
                        for x in all_lines(ch, ind + ws):
                            yield x
                    continue
            if ind:
                line = (' '*ind) + line
            yield line, mNI, i
    #@-others
    return all_lines(ltm.nodes[NI[0]], 0)
#@-others
#@-leo
