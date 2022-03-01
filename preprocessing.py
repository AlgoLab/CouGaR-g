"""All functions that can be applied as preprocessing"""
from src.pipeline import (
    register_in_pipeline, # decorator to make available a function to use with Pipeline class
    Pipeline,
)

@register_in_pipeline
def divide_by_max(npy,):
    "The input npy divided by his maximum value"
    npy /= npy.max()
    return npy

@register_in_pipeline
def divide_by_sum(npy,):
    "The input npy divided by the sum of their values"
    npy /= npy.sum()
    return npy