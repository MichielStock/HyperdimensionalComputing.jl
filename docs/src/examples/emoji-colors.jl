using Handcalcs #hide

# # Introduction
#
# This example explores HDC for a toy regression/classification problem:
# we are going to try to predict the color of a set of emoji's
# based on noisy examples. Then, we will show that HDC can be 
# very robust against superfluous labels, which can be treated as noise
# by the operations. This example also shows how to encode vectors (RGB colors).


# # Generating the data
#
# We start by generating a dataset with some emojis
# and corresponding colors.

using Colors, Random

randcol() = RGB(rand(), rand(), rand());  # generating a random color

# collect all colors
reds = [RGB((c ./ 255)...) for (n, c) in Colors.color_names if occursin("red", n)]
blues = [RGB((c ./ 255)...) for (n, c) in Colors.color_names if occursin("blue", n)]
greens = [RGB((c ./ 255)...) for (n, c) in Colors.color_names if occursin("green", n)]
oranges = [RGB((c ./ 255)...) for (n, c) in Colors.color_names if occursin("orange", n)]
greys = [RGB((c ./ 255)...) for (n, c) in Colors.color_names if occursin("grey", n)]
yellows = [RGB((c ./ 255)...) for (n, c) in Colors.color_names if occursin("yellow", n)]
whites = [RGB((c ./ 255)...) for (n, c) in Colors.color_names if occursin("white", n)]

emojis_colors = Dict(:🚒 => reds, :💦 => blues, :🌱 => greens, :🌅 => oranges, :🐺 => greys, :🍌 => yellows, :🥚 => whites)

emojis = collect(keys(emojis_colors))

# Our first toy data set contain 100 examples of emojis with an associated color

toy_data1 = [rand(emojis) |> l -> (l, rand(emojis_colors[l])) for i in 1:100]

# Our second toy data set is a bit more challenging, we randomly pick 500 emojis 
# and give it **three** color labels. However, only one of them is drawn from
# the true color distribution, the two other are just random.

toy_data2 = [rand(emojis) |> l -> (l, shuffle!([rand(emojis_colors[l]),
    randcol(), randcol()])) for i in 1:500]

# Will we still be able to learn to match the color and emoji, when two of the 
# three color labels are incorrect (and we don't know which one)?

# # Encoding

# Encoding emojis is easy, each one is a random HV. I'll define a small
# function so it is easy to play with different dimensions or HV types.

using HyperdimensionalComputing

hv() = BipolarHV()

emojis_hvs = Dict(s => hv() for s in emojis)

# Colors are a bit more tricky. This is an example color:

acolor = randcol()

acolor.r, acolor.g, acolor.b

md"""
We see that a colour can be represented by three numbers: the fractions of red, green, and blue. Every value is just a number between 0 and 1. If we can construct an embedding for numbers, we can represent a colour as a *binding* of three numbers.

Representing numbers in a fixed interval $[a, b]$ with HDVs is relatively easy. We first divide the interval into $k$ equal parts. Then, we generate an HDV representing the lower bound of the interval. We replace a fraction of $1/k$ of the previous vector for every step with fresh random bits.
"""