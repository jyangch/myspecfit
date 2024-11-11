import numpy as np


class Stat(object):

    def __init__(self):
        self.clear()


    def clear(self):
        self.func = None
        self.expr = None


    def set(self, expr):
        self.func = self.set_stat(expr)
        self.expr = expr


    def set_stat(self, expr):
        if expr == 'chi2': return self.chi2
        elif expr == 'cstat': return self.cstat
        elif expr == 'pgstat': return self.pgstat
        elif expr == 'pgfstat': return self.pgfstat
        else: raise ValueError('invalid value for expr!')


    @property
    def info(self):
        if self.expr is None:
            print('+-----------------------------------------------+')
            print(' The statistic method has not been determined!')
            print('+-----------------------------------------------+\n')
        else:
            print('+-----------------------------------------------+')
            print(' The statistic method: ' + self.expr)
            print('+-----------------------------------------------+\n')


    @staticmethod
    def chi2(S, B, M, ts, tb, serr, berr):
        # S, B are the number of counts respectively in the source and background spectrum.
        # M is the expected rate of events from the source model.
        # ts is the exposure for the source.
        # serr and berr are the error of source and background.

        stat = 0
        for i in range(len(S)):
            si = S[i]
            bi = B[i]
            mi = M[i]
            sierr = serr[i]
            bierr = berr[i]
            
            if tb != 0:
                bi = bi / tb * ts
                bierr = bierr / tb * ts
                
            tierr = np.sqrt(sierr ** 2 + bierr ** 2)
            stat += (si - bi - mi * ts) ** 2 / tierr ** 2
        return stat


    @staticmethod
    def cstat(S, B, M, ts, tb, serr, berr):
        # S, B are the number of counts respectively in the source and background spectrum.
        # M is the expected rate of events from the source model.
        # ts and tb are the exposure for the source and background spectrum.

        FLOOR = 1.0e-5
        stat = 0
        # F = np.zeros_like(S, dtype=float)

        for i in range(len(S)):
            si = S[i]
            bi = B[i]
            mi = M[i]
            tt = ts + tb
            mi = max(mi, FLOOR / ts)

            if si == 0.0:
                stat += ts * mi - bi * np.log(tb / tt)
            else:
                if bi == 0.0:
                    if mi <= si / tt:
                        stat += -tb * mi - si * np.log(ts / tt)
                    else:
                        stat += ts * mi + si * (np.log(si) - np.log(ts * mi) - 1)
                else:
                    # now the main case where both data and background !=0
                    # Solve quadratic equation for f. Use the positive root to ensure
                    # that f > 0.
                    a = tt
                    b = tt * mi - si - bi
                    c = -bi * mi
                    d = np.sqrt(b * b - 4.0 * a * c)
                    # Avoid round-off error problems if b^2 >> 4ac (see eg Num.Recipes)
                    if b >= 0:
                        fi = -2 * c / (b + d)
                    else:
                        fi = -(b - d) / (2 * a)
                    # F[i] = fi
                    # note that at this point f must be > 0 so the log
                    # functions below will be valid.
                    stat += ts * mi + tt * fi - si * np.log(ts * mi + ts * fi) - bi * np.log(tb * fi) \
                            - si * (1 - np.log(si)) - bi * (1 - np.log(bi))
        return 2.0 * stat


    @staticmethod
    def pgstat(S, B, M, ts, tb, serr, berr):
        # S, B are the number of counts respectively in the source and background spectrum.
        # M is the expected rate of events from the source model.
        # ts and tb are the exposure for the source and background spectrum.
        # berr is the error of background.

        FLOOR = 1.0e-5
        stat = 0
        # F = np.zeros_like(S, dtype=float)

        for i in range(len(S)):
            si = S[i]
            bi = B[i]
            mi = M[i]
            bierr = berr[i]
            tr = ts / tb

            # special case for bierr = 0
            if bierr == 0.0:
                mbi = max(mi + bi / tb, FLOOR / ts)
                stat += ts * mbi
                if si > 0.0:
                    stat += si * (np.log(si) - np.log(ts * mbi) - 1)
            else:
                if si == 0.0:
                    stat += ts * mi + bi * tr - 0.5 * (bierr * tr) ** 2
                else:
                    # Solve quadratic equation for fi, using Numerical Recipes technique
                    # to avoid round-off error that can easily cause problems here
                    # when b^2 >> ac.
                    a = tb ** 2
                    b = ts * bierr ** 2 - tb * bi + tb ** 2 * mi
                    c = ts * bierr ** 2 * mi - si * bierr ** 2 - tb * bi * mi
                    if b >= 0.0:
                        sign = 1.0
                    else:
                        sign = -1.0
                    q = -0.5 * (b + sign * np.sqrt(b ** 2 - 4.0 * a * c))
                    fi = q / a
                    if fi < 0.0:
                        fi = c / q
                    # F[i] = fi
                    # note that at this point fi must be > 0 so the log
                    # functions below will be valid.
                    stat += ts * (mi + fi) - si * np.log(ts * mi + ts * fi) + \
                            0.5 * (bi - tb * fi) * (bi - tb * fi) / bierr ** 2 - si * (1 - np.log(si))
                    # if np.isnan(stat) and not np.isnan(mi) and not np.isinf(mi):
                    #    return i
        return 2.0 * stat


    @staticmethod
    def pgfstat(S, B, M, ts, tb, serr, berr):
        # S, B are the number of counts respectively in the source and background spectrum.
        # M is the expected rate of events from the source model.
        # ts and tb are the exposure for the source and background spectrum.
        # berr is the error of background.

        FLOOR = 1.0e-5
        stat = 0

        for i in range(len(S)):
            si = S[i]
            bi = B[i]
            mi = M[i]
            bierr = berr[i]
            tr = ts / tb

            # special case for bierr = 0
            if bierr == 0.0:
                mbi = max(mi + bi / tb, FLOOR / ts)
                stat += ts * mbi
                if si > 0.0:
                    stat += si * (np.log(si) - np.log(ts * mbi) - 1)
            else:
                if si == 0.0:
                    stat += ts * mi + bi * tr - 0.5 * (bierr * tr) ** 2 + 0.5 * np.log(2 * np.pi * bierr ** 2)
                else:
                    # Solve quadratic equation for fi, using Numerical Recipes technique
                    # to avoid round-off error that can easily cause problems here
                    # when b^2 >> ac.
                    a = tb ** 2
                    b = ts * bierr ** 2 - tb * bi + tb ** 2 * mi
                    c = ts * bierr ** 2 * mi - si * bierr ** 2 - tb * bi * mi
                    if b >= 0.0:
                        sign = 1.0
                    else:
                        sign = -1.0
                    q = -0.5 * (b + sign * np.sqrt(b ** 2 - 4.0 * a * c))
                    fi = q / a
                    if fi < 0.0:
                        fi = c / q
                    # note that at this point fi must be > 0 so the log
                    # functions below will be valid.
                    stat += ts * (mi + fi) - si * np.log(ts * mi + ts * fi) + \
                            0.5 * (bi - tb * fi) * (bi - tb * fi) / bierr ** 2 - \
                            si * (1 - np.log(si)) + 0.5 * np.log(2 * np.pi * bierr ** 2)
        return 2.0 * stat
