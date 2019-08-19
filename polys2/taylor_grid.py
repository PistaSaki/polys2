import numpy as np
import tensorflow as tf
import numbers
import itertools as itt
from scipy.special import factorial, binom
from matplotlib import pyplot as pl

from pitf import ptf, nptf
from pitf.ptf import is_tf_object
from . import tensor_utils as pit

from .batch_utils import Batched_Object
from .engine import (array_poly_prod, eval_poly, get_1D_Taylor_matrix,
        get_1d_Taylor_coef_grid, get_1D_integral_of_piecewise_poly,
        get_integral_of_spline_from_taylors_1D,
        get_spline_from_taylors_1D,
        get_Catmul_Rom_Taylors_1D,
        )
from .plot_utils import plot_fun

from .poly import Poly, get_bin_indices, Val_Indexed_Object
from .poly_mesh import PolyMesh

from typing import Union, Tuple, List



    
class TaylorGrid(Batched_Object, Val_Indexed_Object):
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
    
    @property
    def batch_shape(self):
        return self.coef.shape[:self.batch_ndim]
    
    @property
    def val_shape(self):
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
    
    
      