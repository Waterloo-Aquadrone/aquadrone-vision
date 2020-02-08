# aquadrone2020vision

# Overview of Project

## Goals
The vision package is intended to take footage from the Zed camera, identify any objects within it, as well as their location, and send that information to the Software team in the form of: item, x, y, z.

## Code structure

There are a couple of major components to the code.

### Creating Data
splitVideoToFrames
labelFrames
augmentData

### Preprocessing data

### Locating and labelling objects
Model class
Data Class

### Outputting data
translate corners to differentials
look at individual struct by object for number of points
translate number of received points
ideas
- maybe do some kind of edge detection knowing the object?
- force number of corners / shape