
    

    
    


def ipow(base, exponent):
    """ For integral non-negative `exponent` calculate `base ** exponent` using only multiplication.
    """
    result = 1
    
    while exponent > 0:
        if exponent % 2 == 1:
            result *= base

        exponent //= 2
        base *= base
        
    return result

        
class PolyMesh(Batched_Object, TF_Runnable):
    
    def __init__(self, coef, params, batch_ndim = 0,  val_ndim = 0):
        
        self.coef = coef
        self.batch_ndim = batch_ndim
        self.params = [np.array(cc, dtype = nptf.np_dtype(coef)) for cc in params ]
        self.val_ndim = val_ndim
        
        message = (
            f"coef.shape={self.coef.shape}, batch_ndim={batch_ndim}, "
            f"var_ndim = {self.var_ndim}, val_ndim = {self.val_ndim}, "
            f"bins_shape = {self.bins_shape}"
        )
        assert len(self.coef.shape) == self.batch_ndim + 2 * self.var_ndim + self.val_ndim, message
        assert list(self.coef.shape[self.batch_ndim: self.batch_ndim + self.var_ndim]) == list(self.bins_shape), message
    
        
    def __repr__(self):
        s = "Polymesh( " 
        s += "var_ndim = " + str(self.var_ndim)
        s += ", params.lengts = " + str([len(par) for par in self.params]) 
        if self.batch_ndim > 0:
            s += ", batch_ndim = " + str(self.batch_ndim)
        if self.val_ndim > 0:
            s += ", val_ndim = " + str(self.val_ndim)
        s += ", coef.shape = " + str(self.coef.shape) 
        
        s += " )"
            
        
        return s
        
    def __mul__(self, other):
            
        f = self.to_Poly()
        g = other.to_Poly() if isinstance(other, PolyMesh) else other
        return (f * g).to_PolyMesh(self.params)

    def __rmul__(self, other):
        return self * other
    
    def __add__(self, other):
        f = self.to_Poly()
        g = other.to_Poly() if isinstance(other, PolyMesh) else other
        return f.__add__(g).to_PolyMesh(self.params)
        
    def __radd__(self, other):
        return self + other
        
    def __sub__(self, other):
        return self + (-1) * other
    
    def __rsub__(self, other):
        return (-1) * self + other
        
    def __truediv__(self, other):
        return 1/other * self
        
    def __pow__(self, exponent):
        n = exponent
        assert isinstance(n, numbers.Integral)
        assert n > 0
        return ipow(self, exponent)
        
        
    
    def der(self, k = None):
        """Derivative w.r.t. $x_k$.$"""
        
        if k is None:
            if self.var_ndim == 1:
                k = 0
            else:
                raise Exception("You did not specify by which coordinate you want to differentiate. ")    
        
        df = self.to_Poly().der(k).to_PolyMesh(self.params)
        
        ## since the parametrization is different in each bin, we must rescale
        bin_index = self.batch_ndim + k
        scale = self.params[k][1:] - self.params[k][:-1]
        
        sh = [1] * nptf.ndim(self.coef)
        sh[bin_index] = -1
        scale = scale.reshape(sh)
        
        df.coef = df.coef / scale
        
        return df
        
    def _get_batch(self, selector):
        coef = self.coef[selector]
        batch_ndim = self.batch_ndim - (len(self.coef.shape) - len(coef.shape))
        assert batch_ndim >= 0
        return PolyMesh(
            coef = coef,
            params = self.params,
            batch_ndim = batch_ndim, 
            val_ndim = self.val_ndim            
        )
    
    def _get_data_to_execute(self):
        if ptf.is_tf_object(self.coef):
            return self.coef
        else:
            return []

        
    def _make_copy_from_executed_data(self, data):
        if ptf.is_tf_object(self.coef):
            return PolyMesh(
                coef = data,
                params = self.params,
                batch_ndim = self.batch_ndim,
                val_ndim = self.val_ndim,
            )
        else:
            return self
    
    @property
    def batch_shape(self):
        if ptf.is_tf_object(self.coef):
            raise NotImplementedError("Tf objects not implemented yet.")
        else:
            return self.coef.shape[:self.batch_ndim]
    
    @property
    def val_shape(self):
        if ptf.is_tf_object(self.coef):
            raise NotImplementedError("Tf objects not implemented yet.")
        else:
            return self.coef.shape[self.batch_ndim + 2 *self.var_ndim:]
    
    @property
    def var_ndim(self):
        return len(self.params)
    
    @property
    def degs(self):
        return self.coef.shape[self.batch_ndim + self.var_ndim: self.batch_ndim + 2 *self.var_ndim]
    
    @property
    def grid_shape(self):
        return [len(par) for par in self.params]
    
    @property
    def bins_shape(self):
        return [len(par) - 1 for par in self.params]
    
    
    def __call__(self, x):
        if not is_tf_object(x):
            x = np.array(x)
        
        assert nptf.ndim(x) - 1 == self.batch_ndim, (
            "Batch-ndim of `x` is {} "
            "but the batch-ndim of this MeshGrid is {}.".format(
                nptf.ndim(x) - 1, self.batch_ndim
            )
        )
        assert x.shape[-1] == self.var_ndim, (
            "Variable-dimension of this MeshGrid is {} "
            "but the dimension of `x` you want to plug into is is {}.".format(
                self.var_ndim, x.shape[-1], 
            ) 
        )
        
        ## divide `x` into list of its coordinates
        x_unstack = nptf.unstack(x, axis = -1)
        
        ## force all coordinates of `x` into the range of params
        x_unstack = [
            nptf.maximum(cc[0], nptf.minimum(t, cc[-1])) 
            for cc, t in zip(self.params, x_unstack)
        ]
        
        ## find the bin (polynomial patch) containing x
        bin_indices_unstack = [
             nptf.minimum(get_bin_indices(np.array(cc), t), len(cc)-2) 
             for cc, t in zip(self.params, x_unstack)
            ]
        
        bin_indices = nptf.stack( bin_indices_unstack, axis = -1)
        
        poly = Poly(
            coef = nptf.batched_gather_nd(
                a = self.coef, indices = bin_indices, batch_ndim = self.batch_ndim
            ),
            batch_ndim = self.batch_ndim, 
            var_ndim=self.var_ndim, 
            val_ndim=self.val_ndim 
        )
        
        ## reparametrize `x` in the relative coordinates of the corresponding bin
        bin_start = nptf.stack(
            [
             nptf.gather(cc, ii)
             for cc, ii in zip(self.params, bin_indices_unstack)
            ],
            axis = -1
        )
        
        bin_end = nptf.stack(
            [
             nptf.gather(cc, ii + 1)
             for cc, ii in zip(self.params, bin_indices_unstack)
            ],
            axis = -1
        ) 
        
        x_rel = (x - bin_start) / (bin_end - bin_start)
    
        return poly(x_rel)
    
    def to_Poly(self):
        return Poly(
            coef = self.coef, 
            batch_ndim=self.batch_ndim + self.var_ndim, 
            var_ndim=self.var_ndim, 
            val_ndim=self.val_ndim
        )
    
    
    def val_mul(self, a):
        """
        Multiply values by tensor `a` s.t. `a.ndim == self.val_ndim`.
        """
        f = self.to_Poly()
        return f.val_mul(a).to_PolyMesh(self.params)
        
    def val_sum(self, axis = None):
        """
        For a tensor-valued `PolyMesh` return `PolyMesh` whose values are sums of the values of the original.
        """
        f = self.to_Poly()
        return f.val_sum(axis).to_PolyMesh(self.params)
    
    
    def bin_indices(self):
        return itt.product(*[range(len(par) -1) for par in self.params])
    
    def bin_start(self, ii):
        return np.array(
            [par[i] for i, par in zip(ii, self.params) ]
        )
    
    def bin_end(self, ii):
        return np.array(
            [par[i + 1] for i, par in zip(ii, self.params) ]
        )
    
    def domain_start(self):
        return [par[0] for par in self.params]
    
    
            
    def plot(self, **kwargs):
        plot_fun(self, 
            start = [par[0] for par in self.params],
            end = [par[-1] for par in self.params],
            **kwargs
        )
        
    def contour_plot(self, show_labels = True, show_grid = True, **kwargs):
        xxx, yyy = [np.linspace(par[0], par[-1], 50) for par in self.params] 
        fff = np.array([[ self([x,y]) for y in yyy ] for x in xxx]).T
        cp = pl.contour( xxx, yyy, fff, **kwargs)
        
        if show_labels:
            pl.clabel(cp)
        
        if show_grid:
            pl.scatter(*zip(*itt.product(*self.params)), marker = "+")
        
        return cp
            
    
    def integrate(self):
        coef = self.coef
        for i in range(self.var_ndim):
            coef = get_1D_integral_of_piecewise_poly(
                coef = coef, 
                bin_axis = self.batch_ndim, 
                polynom_axis = self.batch_ndim + self.var_ndim - i , 
                control_times = self.params[i]
            )
            
        return coef
    
    
class TaylorGrid(Batched_Object, Val_Indexed_Object, TF_Runnable):
    def __init__(self, coef, params, batch_ndim = 0,  val_ndim = 0):
        self.coef = coef
        self.batch_ndim = batch_ndim
        self.params = params
        self.val_ndim = val_ndim
        
        assert len(self.coef.shape) == self.batch_ndim + 2 * self.var_ndim + self.val_ndim
        
        grid_shape_from_coef = list(self.coef.shape[self.batch_ndim: self.batch_ndim + self.var_ndim])
        grid_shape_from_params = list(self.grid_shape)
        assert grid_shape_from_coef == grid_shape_from_params, (
            "The coeficients suggest grid shape " + str(grid_shape_from_coef) +
            " while the one inferred from params is " + str(grid_shape_from_params)
        )
    
    def __repr__(self):
        s = "TaylorGrid( " 
        s += "var_ndim = " + str(self.var_ndim)
        s += ", params.lengts = " + str([len(par) for par in self.params]) 
        if self.batch_ndim > 0:
            s += ", batch_ndim = " + str(self.batch_ndim)
        if self.val_ndim > 0:
            s += ", val_ndim = " + str(self.val_ndim)
        s += ", coef.shape = " + str(self.coef.shape) 
        
        s += " )"
            
        return s
        
    def _get_batch(self, selector):
        coef = self.coef[selector]
        batch_ndim = self.batch_ndim - (len(self.coef.shape) - len(coef.shape))
        assert batch_ndim >= 0
        return TaylorGrid(
            coef = coef,
            params = self.params,
            batch_ndim = batch_ndim, 
            val_ndim = self.val_ndim            
        )
        
    def _val_getitem(self, selector):
        if not isinstance(selector, tuple):
            selector = (selector,)
            
        coef = self.coef[
            (slice(None), )*(self.batch_ndim + 2 * self.var_ndim)
            + selector
        ]
        
        val_ndim = self.val_ndim - (len(self.coef.shape) - len(coef.shape))
        assert val_ndim >= 0, "val_ndim = " + str(val_ndim)
        return TaylorGrid(
            coef = coef, 
            params = self.params,
            batch_ndim = self.batch_ndim, 
            val_ndim = val_ndim            
        )
    
    def _get_data_to_execute(self):
        if ptf.is_tf_object(self.coef):
            return self.coef
        else:
            return []

        
    def _make_copy_from_executed_data(self, data):
        if ptf.is_tf_object(self.coef):
            return TaylorGrid(
                coef = data,
                params = self.params,
                batch_ndim = self.batch_ndim,
                val_ndim = self.val_ndim,
            )
        else:
            return self
    
    @property
    def batch_shape(self):
        if ptf.is_tf_object(self.coef):
            raise NotImplementedError("Tf objects not implemented yet.")
        else:
            return self.coef.shape[:self.batch_ndim]
    
    @property
    def val_shape(self):
        if ptf.is_tf_object(self.coef):
            raise NotImplementedError("Tf objects not implemented yet.")
        else:
            return self.coef.shape[self.batch_ndim + 2 *self.var_ndim:]
    
    @property
    def var_ndim(self):
        return len(self.params)
    
    @property
    def degs(self):
        return self.coef.shape[self.batch_ndim + self.var_ndim: self.batch_ndim + 2 *self.var_ndim]
    
    @property
    def grid_shape(self):
        return [len(par) for par in self.params]
    
    @property
    def bins_shape(self):
        return [len(par) - 1 for par in self.params]
    
    def __mul__(self, other, truncation = None):
        if truncation is None:
            truncation = np.minimum(self.degs, getattr(other, "degs", 0))
            
        f = self.to_Poly()
        g = other.to_Poly() if isinstance(other, TaylorGrid) else other
        return f.__mul__(g, truncation=truncation).to_TaylorGrid(self.params)

    def __rmul__(self, other):
        return self * other
    
    def __add__(self, other):
        f = self.to_Poly()
        g = other.to_Poly() if isinstance(other, TaylorGrid) else other
        return f.__add__(g).to_TaylorGrid(self.params)
        
    def __radd__(self, other):
        return self + other
        
    def __sub__(self, other):
        return self + (-1) * other
    
    def __rsub__(self, other):
        return (-1) * self + other
        
        
    def __truediv__(self, other):
        return 1/other * self
        
    def __rtruediv__(self, other):
        return self.truncated_inverse() * other
        
    def __pow__(self, other):
        return self.truncated_power(other)
        
    def truncate_degs(self, limit_degs):
        f = self.to_Poly()
        return f.truncate_degs(limit_degs).to_TaylorGrid(self.params)
    
    def pad_degs(self, minimal_degs):
        f = self.to_Poly()
        return f.pad_degs(minimal_degs).to_TaylorGrid(self.params)
        
    def val_mul(self, a):
        """
        Multiply values by tensor `a` s.t. `a.ndim == self.val_ndim`.
        """
        f = self.to_Poly()
        return f.val_mul(a).to_TaylorGrid(self.params)
        
    def val_sum(self, axis = None):
        """
        For a tensor-valued `TaylorGrid` return `TaylorGrid` whose values are sums of the values of the original.
        """
        f = self.to_Poly()
        return f.val_sum(axis).to_TaylorGrid(self.params)
        
        
    
    def to_Poly(self):
        return Poly(
            coef = self.coef, 
            batch_ndim=self.batch_ndim + self.var_ndim, 
            var_ndim=self.var_ndim, 
            val_ndim=self.val_ndim
        )
    
    def get_poly_at_node(self, ii):
        ii = tuple([None] * self.batch_ndim + list(ii))
        return Poly(coef = self.coef[ii], batch_ndim = self.batch_ndim, var_ndim=self.var_ndim, val_ndim=self.val_ndim )
    
    def get_node(self, ii):
        return np.array([par[i] for i, par in zip(ii, self.params) ])
    
    def grid_indices(self):
        return itt.product(*[range(len(par)) for par in self.params])
    
    
    def plot(self, **kwargs):
        for ii in self.grid_indices():
            this_node = self.get_node(ii)
            previous_node = self.get_node([max(0, i-1) for i in ii])
            next_node = self.get_node([min(i+1, len(par) - 1) for i,par in zip(ii, self.params)])
            
            poly = self.get_poly_at_node(ii)
            poly = poly.shift(this_node)
            poly.plot(
                start = 2/3 * this_node + 1/3 * previous_node,
                end = 2/3 * this_node + 1/3 * next_node,
                **kwargs
            )
            
    def get_spline(self):
        coef = self.coef
        
        for i, par in enumerate(self.params):
            coef = get_spline_from_taylors_1D(
                taylor_grid_coeffs = coef, 
                bin_axis = self.batch_ndim + i, 
                polynom_axis = self.batch_ndim + self.var_ndim + i, 
                control_times = par
            )
            
        return PolyMesh(
            coef = coef, params = self.params, batch_ndim=self.batch_ndim, val_ndim=self.val_ndim
        )
        
    def integrate_spline(self):
        coef = self.coef
        for i in range(self.var_ndim):
            coef = get_integral_of_spline_from_taylors_1D(
                taylor_grid_coeffs= coef, 
                bin_axis = self.batch_ndim, 
                polynom_axis = self.batch_ndim + self.var_ndim - i , 
                control_times = self.params[i]
            )
            
        return coef

    
    def truncated_exp(self, degs = None):
        f = self.to_Poly()
        return f.truncated_exp(degs).to_TaylorGrid(self.params)
    
    def truncated_inverse(self, degs = None):
        f = self.to_Poly()
        return f.truncated_inverse(degs).to_TaylorGrid(self.params)
    
    def truncated_power(self, exponent, degs = None):
        """Return `self` raised to the exponent (possibly real) truncated to `degs`. 
        """
        f = self.to_Poly()
        return f.truncated_power(
                    exponent = exponent, degs = degs
            ).to_TaylorGrid(self.params)
        
    
    
    @staticmethod
    def from_Catmull_Rom(coef, params, batch_ndim = 0, val_ndim = None):
        if val_ndim is None:
            val_ndim = len(coef.shape) - batch_ndim - len(params)
            
        for i, par in enumerate(params):
            coef = get_Catmul_Rom_Taylors_1D(
                coef = coef,
                control_index = batch_ndim + i,
                control_times = par,
                added_index = batch_ndim + len(params) + i,
            )
            
        return TaylorGrid(coef, params, batch_ndim= batch_ndim, val_ndim=val_ndim)
    
    @staticmethod
    def from_Gauss_pdf(params, mu, K, batch_ndim = None, var_ndim = None):
        if batch_ndim is None:
            batch_ndim = nptf.ndim(mu) - 1

        if var_ndim is None:
            var_ndim = int(mu.shape[-1])

        assert (
            (batch_ndim == nptf.ndim(mu) - 1 == nptf.ndim(K) - 2) and
            (var_ndim == mu.shape[-1] == K.shape[-1] == K.shape[-2] )
        ),("Problem with dimensions: mu.shape =" + str(mu.shape) + 
           ", K.shape = " + str(K.shape) + ", batch_ndim = " + str(batch_ndim) +
           ", var_ndim = " + str(var_ndim)
        )

        exponent = -1/2 * (
            Poly.from_tensor(K, batch_ndim=batch_ndim) + 
            (-2) * Poly.from_tensor(
                (K @ mu[..., None])[..., 0], 
                batch_ndim=batch_ndim,
                var_ndim = var_ndim
            )  +
            Poly.from_tensor(
                (mu[..., None,:] @ K @ mu[..., None])[...,0, 0], 
                batch_ndim=batch_ndim,
                var_ndim = var_ndim
            )     
        )

        tg = exponent.get_Taylor_grid(params = params, truncs = 2)
        tg = tg.truncated_exp()

        const = (2 * np.pi) ** (-tg.var_ndim / 2) * nptf.det(K)**(1/2)
        tg *= const

        return tg
    
    
      