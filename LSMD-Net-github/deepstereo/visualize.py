from visdom import Visdom

class TrainerVisdom:
    def __init__(self, stage, **kws):
        if stage == 'fit':
            self.viz = Visdom(**kws)
        else:
            self.viz = None

    def prep_win(self, win, title=None, **kws):
        if not self.viz:
            return

        if not title:
            title = win

        opts = kws
        opts['title'] = title
        return {
            'win': win,
            'opts': opts,
        }

    def __bool__(self):
        return bool(self.viz)

    def images(self, t, win, title=None, **kws):
        kws = self.prep_win(win, title, **kws)
        if kws:
            self.viz.images(t, **kws)

    def text(self, msg, win, title=None, **kws):
        kws = self.prep_win(win, title, **kws)
        if kws:
            self.viz.text(msg, **kws)
