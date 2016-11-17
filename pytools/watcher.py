class Watcher:
    def __init__(self, watch_list):
        self.__watch_list = watch_list

    def get_watch_list(self):
        wl = {}
        for ds, ev in self.__watch_list:
            wl[ds.get_tag()] = ev.get_tag()
        return wl

    def __str__(self):
        return str(self.get_watch_list())
