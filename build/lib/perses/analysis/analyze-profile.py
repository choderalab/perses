#!/usr/bin/env python

import pstats
p = pstats.Stats('profile.out')
p.strip_dirs().sort_stats(-1).print_stats()

nreport = 100

p.sort_stats('cumulative').print_stats(nreport)
#p.sort_stats('time').print_stats(nreport)
