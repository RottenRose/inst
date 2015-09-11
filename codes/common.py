import string
import os
import numpy as np
import sys


class Dims(object):
    """
    input: str or list or tuple or number
    such as:
    '1,2,3,4'
    (1,2,3,4)
    [1,2,3,4]
    1 2 3 4
    """
    def __init__(self, *t):
        self.batch_, self.channel_, self.y_, self.x_ = 0,0,0,0
        self.count_ = self.batch_*self.channel_ * self.y_*self.x_
        if len(t) == 0:
            self.batch_, self.channel_, self.y_, self.x_ = 0,0,0,0
            self.count_ = self.batch_*self.channel_ * self.y_*self.x_
            return
        if len(t) > 1:
            self.batch_, self.channel_, self.y_, self.x_ = t
            self.count_ = self.batch_*self.channel_ * self.y_*self.x_
            return
        if len(t) == 1:
            if type(t[0]) == tuple or type(t[0]) == list:
                self.batch_, self.channel_, self.y_, self.x_ = t[0]
                self.count_ = self.batch_*self.channel_ * self.y_*self.x_
                return
            if type(t) == str and ',' in t:
                self.batch_, self.channel_, self.y_, self.x_ = [int(x) for x in t.split(',') if len(x) >0]
                self.count_ = self.batch_*self.channel_ * self.y_*self.x_
                return

    def print_info(self):
        print '\t%d\t%d\t%d\t%d' % (self.batch_, self.channel_, self.y_, self.x_),


class Atom(object):
    def __init__(self, code, meaning):
        self.code = code
        self.meaning = meaning

    def print_info(self):
        print '%s code: %s' % (self.meaning, self.code)