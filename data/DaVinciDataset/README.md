# Leonardo da Vinci's Sketch Dataset

This is the dataset release that accompanies the paper ["Learning to Restore Deteriorated Line Drawing"](http://hi.cs.waseda.ac.jp/~iizuka/projects/line_restoration/).

If you use this data, please cite it as:
```
@Article{SasakiCGI2018,
   author    = {Kazuma Sasaki and Satoshi Iizuka and Edgar Simo-Serra and Hiroshi Ishikawa},
   title     = {{Learning to Restore Deteriorated Line Drawing}},
   journal   = {The Visual Computer (Proc. of Computer Graphics International 2018)},
   year      = {2018},
   volume    = {34},
   number    = {6-8},
   pages     = {1077-1085},
}
```

## Overview

We collect 71 of Leonard da Vinci's old sketch scans, and manually provide annotations of the underlying line drawing. The annotations are added by drawing on top of the old sketches using a pen tablet. Here, we choose those sketches with limited amount of shading that can be considered line drawings, avoiding artistic drawings and the most heavily deteriorated sketches. We do not annotate dirt nor text as ground truth lines. Out of the 71 images, we reserve 10 for evaluation, leaving 61 images to be used for training.

## Layout

The images are split into two folders, one is the training set and the other is the test set, which are used in the paper. For each of the image pairs, `da#.png` are the original deteriorated sketches and `da#_gt.png` are the ground truth line annotations, where `#` denotes the image id.

```
README.md # This readme
train/    # Training images
test/     # Test images

```

## License
```
  Copyright (C) <2018> <Kazuma Sasaki, Satoshi Iizuka, Edgar Simo-Serra, Hiroshi Ishikawa>

  This work is licensed under the Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy
  of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or
  send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
```

