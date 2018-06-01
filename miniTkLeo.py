#@+leo-ver=5-thin
#@+node:vitalije.20180514194301.1: * @file miniTkLeo.py
#@@language python
# pylint: disable=invalid-name
# pylint: disable=no-member
G = None
bridge = True
#@+<< imports >>
#@+node:vitalije.20180518115138.1: ** << imports >>
import datetime
import cProfile
import pstats
from tkinter import Canvas, Tk, scrolledtext, PanedWindow, PhotoImage, IntVar
from tkinter import font
import threading
import queue
import time
#from tkinter import ttk
from leoDataModel import (
         LeoTreeModel,
         loadLeo,
         loadExternalFiles,
         ltm_from_derived_file,
         p_to_lines,
         paths
    )
import sys
import os
from box_images import boxKW, plusnode, minusnode
assert ltm_from_derived_file
assert p_to_lines
assert paths
#@-<< imports >>
profile_load = False
profile_redraw = True

#@+others
#@+node:vitalije.20180515103819.1: ** class bunch
class bunch:
    def __init__(self, **kw):
        self.__dict__.update(kw)

#@+node:vitalije.20180518132658.1: ** click_h
def click_h(j):
    def select_p(x):
        ltm = G.ltm
        p = ltm.visible_positions[j + G.topIndex.get()]
        i = ltm.positions.index(p)
        gnx = ltm.nodes[i]
        ltm.selectedPosition = p
        G.body.setBodyTextSilently(ltm.attrs[gnx][1])
        draw_tree(G.tree, ltm)
        ltm.invalidate_visual()
    return select_p
#@+node:vitalije.20180518132651.1: ** click_pmicon
def click_pmicon(j):
    def switchExpand(x):
        ltm = G.ltm
        p = ltm.visible_positions[j + G.topIndex.get()]
        if p in ltm.expanded:
            ltm.expanded.remove(p)
            ltm.selectedPosition = p
            i = ltm.selectedIndex
            bstr = ltm.attrs[ltm.nodes[i]][1] if i > -1 else ''
            G.body.setBodyTextSilently(bstr)
        else:
            ltm.expanded.add(p)
        draw_tree(G.tree, ltm)
        ltm.invalidate_visual()
    return switchExpand
#@+node:vitalije.20180515153501.1: ** connect_handlers
def connect_handlers():
    g = G.g
    bw = G.body
    tree = G.tree
    ltm = G.ltm
    topIndex = G.topIndex
    def speedtest(x):
        import timeit
        pos = ltm.positions[-4]
        def f1():
            ltm.promote(pos)
            draw_tree(tree, ltm)
            ltm.promote_children(pos)
            draw_tree(tree, ltm)
        def f2():
            ltm.move_node_down(pos)
            draw_tree(tree, ltm)
            ltm.move_node_up(pos)
            draw_tree(tree, ltm)

        t1 = timeit.timeit(f1, number=100)/100*1000
        t2 = timeit.timeit(f2, number=100)/100*1000
        G.log.insert('end', 'demote/promote average: %.1fms\n'%t1)
        G.log.insert('end', 'up/down average: %.1fms\n'%t2)

    #@+others
    #@+node:vitalije.20180516131732.1: *3* prev_node
    def prev_node(x):
        gnx = ltm.select_prev_node()
        if gnx:
            bw.replace('1.0', 'end', ltm.attrs[gnx][1])
        tree.focus_set()
        topIndex_write(0,0,0)
        return 'break'
    #@+node:vitalije.20180516131735.1: *3* next_node
    def next_node(x):
        gnx = ltm.select_next_node()
        if gnx:
            bw.replace('1.0', 'end', ltm.attrs[gnx][1])
        tree.focus_set()
        topIndex_write(0,0,0)
        return 'break'

    #@+node:vitalije.20180516131739.1: *3* on_body_change
    def on_body_change(x):
        s = bw.get('1.0', 'end - 1c')
        bw.edit_modified(False)
        ltm.body_change(s)
        draw_tree(tree, ltm)
    #@+node:vitalije.20180516131743.1: *3* alt_left
    def alt_left(x):
        gnx = ltm.select_node_left()
        if gnx:
            bw.replace('1.0', 'end', ltm.attrs[gnx][1])
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
        tree.focus_set()
        topIndex_write(0,0,0)

    #@+node:vitalije.20180516131747.1: *3* alt_right
    def alt_right(x):
        gnx = ltm.select_node_right()
        if gnx:
            bw.replace('1.0', 'end', ltm.attrs[gnx][1])
        draw_tree(tree, ltm)
        tree.focus_set()
        topIndex_write(0,0,0)
    #@+node:vitalije.20180516131751.1: *3* mouse_wheel
    def mouse_wheel(ev):
        if ev.num == 4 or ev.delta < 0:
            topIndex.set(max(0, topIndex.get() - 1))
        elif ev.num == 5 or ev.delta > 0:
            HR = 24; nr = rows_count(HR)
            ti = topIndex.get()
            cnt = len(tuple(ltm.display_items(ti, nr)))
            if cnt < nr - 1:
                return
            topIndex.set(topIndex.get() + 1)

    #@+node:vitalije.20180516131800.1: *3* topIndex_write
    def topIndex_write(a, b, c):
        return draw_tree(tree, ltm)

    def show_sel():
        sel = ltm.visible_positions.index(ltm.selectedPosition) + 1
        nr = rows_count(24)
        i = topIndex.get()
        if nr + i < sel or i > sel - 1:
            topIndex.set(max(0, sel - nr // 2))
        G.app.after(30, show_sel)

    show_sel()
    #@+node:vitalije.20180518095655.1: *3* promote_sel
    def promote_sel(x):
        ltm.promote(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
    #@+node:vitalije.20180518095658.1: *3* pr_sel_ind
    def pr_sel_ind(x):
        print(ltm.selectedIndex)
    #@+node:vitalije.20180518095702.1: *3* demote_sel
    def demote_sel(x):
        ltm.promote_children(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
    #@+node:vitalije.20180518095704.1: *3* move_right
    def move_right(x):
        ltm.indent_node(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
        return 'break'
    #@+node:vitalije.20180518095708.1: *3* move_left
    def move_left(x):
        ltm.dedent_node(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
        return 'break'
    #@+node:vitalije.20180518095711.1: *3* move_up
    def move_up(x):
        ltm.move_node_up(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
        return 'break'
    #@+node:vitalije.20180518095714.1: *3* move_down
    def move_down(x):
        ltm.move_node_down(ltm.selectedPosition)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
        return 'break'
    #@+node:ekr.20180531160419.1: *3* traverse_speed
    def traverse_speed(x):
        '''Traverse the entire tree.'''
        g.cls()
        if profile_redraw:
            cProfile.runctx('traverse_speed_helper()',
                globals(), locals(), 'profile_stats')
            print('===== starting profile_stats')
            pattern = r'(miniTkLeo|leo.*)\.py'
            p = pstats.Stats('profile_stats')
            p.strip_dirs().sort_stats('tottime').print_stats(pattern, 50)
        else:
            traverse_speed_helper()
            
    def traverse_speed_helper():
        
        if bridge:
            c = G.c
            c._currentPosition = p = c.rootPosition()
            # c.selectPosition()
        else:
            ltm.selectedPosition = ltm.positions[1]
            ltm.expanded.clear()
            ltm.invalidate_visual()
        draw_tree(tree, ltm)

        def tf(i):
            if 1: # Do everything now.
                n_positions = len(ltm.positions)
                n = min(1000, n_positions)
                if bridge:
                    for i in range(n):
                        p.moveToThreadNext()
                        c._currentPosition = p
                        draw_tree(tree, ltm)
                else:
                    for i in range(n):
                        ltm.select_node_right()
                        draw_tree(tree, ltm)
            else:
                alt_right(None)
                if i < 500 and i < n_positions:
                    tree.after_idle(tf, i + 1)
            dt = datetime.datetime.utcnow() - t1
            t = dt.seconds + 1e-6*dt.microseconds
            G.log.insert('end -1 ch', '%s iterations in %6.2f sec\n'%(n,t))

        t1 = datetime.datetime.utcnow()
        tf(1)

    G.app.bind_all('<Shift-F10>', traverse_speed)
    #@-others
    def expand_all(x):
        ltm.expanded.update(ltm.positions)
        draw_tree(tree, ltm)
        ltm.invalidate_visual()
    topIndex.trace_add('write', topIndex_write)
    bw.bind(g.angleBrackets('TextModified'), on_body_change)
    bw.bind('<Alt-Key-Up>', prev_node)
    bw.bind('<Alt-Key-Down>', next_node)
    bw.bind('<Alt-Key-t>', lambda x:tree.focus_set())
    bw.bind('<Alt-Key-Left>', alt_left)
    bw.bind('<Alt-Key-Right>', alt_right)
    bw.bind('<Alt-x>', pr_sel_ind) # for debugging purposes
    bw.bind_all('<Control-braceright>', promote_sel, add=True)
    bw.bind_all('<Control-braceleft>', demote_sel, add=True)
    bw.bind_all('<F12>', expand_all)
    tree.bind('<Alt-Key-b>', lambda x:bw.focus_set())
    tree.bind('<Key-Return>', lambda x:bw.focus_set())
    tree.bind('<Key-Up>', prev_node)
    tree.bind('<Key-Down>', next_node)
    tree.bind('<Key-Left>', alt_left)
    tree.bind('<Shift-Left>', move_left)
    tree.bind('<Shift-Right>', move_right)
    tree.bind('<Shift-Up>', move_up)
    tree.bind('<Shift-Down>', move_down)
    tree.bind('<Key-Right>', alt_right)
    tree.bind('<MouseWheel>', mouse_wheel)
    tree.bind('<Button-4>', mouse_wheel)
    tree.bind('<Button-5>', mouse_wheel)
    G.app.bind_all('<F10>', speedtest)

    topIndex.trace_add('write', topIndex_write)
#@+node:vitalije.20180515103828.1: ** draw_tree & bridge_items
def draw_tree(canv, ltm):
    '''Redraw the entire visible part of the tree.'''
    g = G.g
    assert g
    #@+<< define bridge_items >>
    #@+node:ekr.20180601045054.1: *3* << define bridge_items >>
    def bridge_items(skip=0, count=None):
        '''
        A generator yielding tuples for visible, non-skipped items:
        
        (pos, gnx, h, levels[i], plusMinusIcon, iconVal, selInd == i)
        '''
        # g.trace('skip', skip, 'count', count)
        c = G.c
        selInd = ltm.selectedIndex
        # print('bridge_items: selInd: %s, skip: %s, count: %s' % (selInd, skip, count))
        i = 1
        p = c.rootPosition()
        while p and (count is None or count > 0):
            # g.trace(p.h)
            if skip > 0:
                skip -= 1
            else:
                # There is one less line to be drawn.
                if count is not None:
                    count -= 1
                # Compute the iconVal for the icon box.
                iconVal = 1 if p.b else 0
                if p.isMarked(): iconVal += 2
                if p.isCloned(): iconVal += 4
                # Compute the +- icon if the node has children.
                if p.hasChildren():
                    plusMinusIcon = 'minus' if p.isExpanded() else 'plus'
                else:
                    plusMinusIcon = 'none'
                # Yield a tuple describing the line to be drawn.
                yield p, p.gnx, p.h, p.level()+1, plusMinusIcon, iconVal, selInd == i
            i += 1
            p.moveToVisNext(c)
    #@-<< define bridge_items >>
        # This binds ltm in bridge_items.
    display_items = bridge_items if bridge else ltm.display_items
    HR = 24 # pixels/per row
    LW = 2 * HR
    count = rows_count(HR)
        # The number of rows.
    items = list(canv.find('all'))
    if len(items) < 1:
        # add selection highliter the first time we draw the tree.
        items.append(
            canv.create_rectangle((0, -100, 300, -100+HR), 
            fill='#77cccc'))
    #
    # The main drawing loop.
    # Each row consists of 3 items: the checkbox, icon and text.
    #
    i, j = 1, 0 # The global row.
    for j, dd in enumerate(display_items(skip=G.topIndex.get(), count=count)):
        p, gnx, h, lev, pm, iconVal, sel = dd
            # The tuples yielded from display_items.
        plusMinusIcon = getattr(G.icons, pm)
        #
        # Update the counts & x,y offsets.
        i = j * 3 + 1
        x = lev * LW - 20
        y = j * HR + HR + 2
        #
        # Set the color
        if sel:
            canv.coords(items[0], 0, y - HR/2 - 2, canv.winfo_width(), y + HR/2 + 2)
            fg = '#000000'
        else:
            fg = '#a0a070'
        #
        # Update or add the items.
        if i + 2 < len(items):
            # The row exists.  Update the items.
            if plusMinusIcon:
                canv.itemconfigure(items[i], image=plusMinusIcon)
                canv.coords(items[i], x, y)
            else:
                canv.coords(items[i], -200, y)
            canv.itemconfigure(items[i + 1], image=G.boxes[iconVal])
            canv.coords(items[i + 1], x + 20, y)
            canv.itemconfigure(items[i + 2], text=h, fill=fg)
            canv.coords(items[i + 2], x + 40, y)
        else:
            # Add 3 more items to canvas.
            items.append(canv.create_image(x, y, image=plusMinusIcon))
            items.append(canv.create_image(x + 20, y, image=G.boxes[iconVal]))
            items.append(canv.create_text(x + 40, y, text=h, anchor="w", fill=fg))
            # Bind clicks to click handlers.
            canv.tag_bind(items[i], '<Button-1>', click_pmicon(j), add=False)
            canv.tag_bind(items[i + 1], '<Button-1>', click_h(j), add=False)
            canv.tag_bind(items[i + 2], '<Button-1>', click_h(j), add=False)
    # Hide any extra item on canvas 
    for item in items[i + 3:]:
        canv.coords(item, 0, -200)
    # g.trace('%s nodes' % (j+1))
#@+node:vitalije.20180514223632.1: ** main
def main(fname):
    #@+others
    #@+node:vitalije.20180518114847.1: *3* create_app
    def create_app():
        app = Tk()
        app.columnconfigure(0, weight=1)
        app.rowconfigure(0, weight=1)
        # Adjust fonts
        font.nametofont('TkFixedFont').config(size=18)
        font.nametofont('TkTextFont').config(size=18)
        font.nametofont('TkDefaultFont').config(size=18)
        return app
    #@+node:vitalije.20180518114953.1: *3* create_gui
    def create_gui(app):
        
        f1 = PanedWindow(app,
            orient='horizontal',
            width=800, height=600,
            sashrelief='ridge',
            sashwidth=4,
        )
        f1.grid(row=0, column=0, sticky="nesw", )
        f2 = PanedWindow(f1, orient='vertical')
        canvW = Canvas(f2, bg='#113333')
        f2.add(canvW)
        logW = scrolledtext.ScrolledText(f2, bg='#223399', fg='#cccc99',
            font=font.nametofont('TkDefaultFont'))
        f2.add(logW)
        bodyW = makeBodyW(f1)
        f1.add(f2)
        f1.add(bodyW)
        return f1, f2, bodyW, canvW, logW
    #@+node:vitalije.20180518114840.1: *3* load_xml
    def load_xml(fname):
        ltm = loadLeo(fname)
        ltm.selectedPosition = ltm.positions[1]
        ltmbytes = ltm.tobytes()
        return ltm, ltmbytes
    #@+node:vitalije.20180518115038.1: *3* start_thread & helper functions
    def start_thread(app, f1, f2, ltmbytes):
        
        #@+others
        #@+node:ekr.20180530110733.1: *4* f_later
        def f_later():
            f1.sash_place(0, 270, 1)
            f2.sash_place(0, 1, 350)
            app.geometry("800x600+120+50")
            app.wm_title(fname)
            app.after_idle(update_model)
            
        #@+node:ekr.20180530110731.1: *4* loadex (collects stats)
        def loadex():
            '''The target of threading.Thread.'''
            if profile_load: # Profile the code.
                cProfile.runctx('loadex_helper()',
                    globals(),
                    locals(),
                    'profile_stats', # 'profile-%s.out' % process_name
                )
                print('===== writing profile_stats')
                p = pstats.Stats('profile_stats')
                p.strip_dirs().sort_stats('tottime').print_stats(50)
                    # .print_stats('leoDataModel.py', 50)
            else:
                loadex_helper()
            
        def loadex_helper():
            ltm2 = LeoTreeModel.frombytes(ltmbytes)
            loaddir = os.path.dirname(fname)
            loadExternalFiles(ltm2, loaddir)
            G.q.put(ltm2)
        #@+node:ekr.20180530110733.2: *4* update_model
        def update_model():
            try:
                m = G.q.get(False)
                ltm.positions = m.positions
                ltm.nodes = m.nodes
                ltm.parPos = m.parPos
                ltm.levels = m.levels
                ltm.attrs = m.attrs
                ltm.gnx2pos = m.gnx2pos
                draw_tree(G.tree, ltm)
                tend = time.monotonic()
                t1 = (tend - tstart)
                if bridge:
                    logW.insert('end', '***Bridge loaded***\n')
                logW.insert('end', 'External files loaded in %.3fs\n'%t1)
            except queue.Empty:
                app.after(100, update_model)
        #@-others
        threading.Thread(target=loadex, name='externals-loader').start()
        app.after_idle(f_later)
    #@-others
    c = None
    if bridge:
        import leo.core.leoBridge as leoBridge
        controller = leoBridge.controller(gui='nullGui',
            loadPlugins=False,
            readSettings=False,
            silent=False, # Print signons, so we know why loading is slow.  
            verbose=False,
        )
        g = controller.globals()
        c = controller.openLeoFile(fname)
        g.trace(c)
    else:
        import leo.core.leoGlobals as g
        g.cls()
    #
    tstart = time.monotonic()
    ltm, ltmbytes = load_xml(fname)
    app = create_app()
    f1, f2, bodyW, canvW, logW = create_gui(app)
    start_thread(app, f1, f2, ltmbytes)
    return bunch(
        c=c, # EKR
        g=g, # EKR
        ltm=ltm,
        app=app,
        tree=canvW,
        body=bodyW,
        log=logW,
        q=queue.Queue(1),
        topIndex=IntVar())
#@+node:vitalije.20180515145134.1: ** makeBodyW
def makeBodyW(parent):
    bw = scrolledtext.ScrolledText(parent, font='Courier 18')
    bw._orig = bw._w + '_orig'
    bw.tk.call('rename', bw._w, bw._orig)
    def proxycmd(cmd, *args):
        result = bw.tk.call(bw._orig, cmd, *args)
        if cmd in ('insert', 'delete', 'replace'):
            bw.event_generate('<<'
                'TextModified>>')
        return result
    bw.tk.createcommand(bw._w, proxycmd)
    bw.setBodyTextSilently = lambda x:bw.tk.call(bw._orig, 'replace', '1.0', 'end', x)
    return bw
#@+node:vitalije.20180515103823.1: ** mk_icons
def mk_icons():
    G.icons = bunch(none=None)
    G.icons.plus = PhotoImage(**plusnode)
    G.icons.minus = PhotoImage(**minusnode)
    G.boxes = [PhotoImage(**boxKW(i)) for i in range(16)]

#@+node:vitalije.20180518115240.1: ** rows_count
def rows_count(h):
    return max(1, G.tree.winfo_height() // h)
#@-others
if __name__ == '__main__':
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = "c:/leo.repo/leo-editor/leo/core/leoPy.leo"
        if not os.path.exists(fname):
            print('not found: %s' % fname)
            print('usage: python miniTkLeo.py <leo document>')
            sys.exit()
    G = main(fname)
        # G is a bunch: app, body, log, q, topIndex (a Tk intVar)
    connect_handlers()
    mk_icons()
    G.tree.after_idle(draw_tree, G.tree, G.ltm)
    G.tree.after_idle(G.tree.focus_set)
    G.tree.bind('<Expose>', lambda x: draw_tree(G.tree, G.ltm), True)
    G.app.mainloop()
#@-leo
