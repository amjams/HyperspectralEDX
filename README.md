# Automated analysis of ultrastructure through large-scale hyperspectral electron microscopy
This is the code and data repository of the paper “Automated analysis of ultrastructure through large-scale hyperspectral electron microscopy”, submitted for review. Here, you can find links to view the full EM maps accompanying the paper, code to reproduce the experiments, and downloadable EDX-EM data for reuse.

[Published paper](https://www.nature.com/articles/s44303-024-00059-7)

<div align="center">
  <img src="https://github.com/amjams/HyperspectralEDX/blob/main/data/HAADFtoColor_zoom.gif" alt="HAADF to Color Dissolve GIF">
</div>

Index to the downloadable data
---------
[nanotomy.org](http://www.nanotomy.org/OA/Duinkerken2024NPJI/)

Update: the viewable maps can also be found on the [Image Data Repository](https://idr.openmicroscopy.org/webclient/?show=project-3102).

Links to viewable EM maps
---------
[Large-scale ColorEM (Figure 1)](http://www.nanotomy.org/avivator/?image_url=http://www.nanotomy.org/OA/Duinkerken2024NPJI/figures/fig1/Figure1_Multichannel.ome.tif)

[HAADF and multi-channel abundance maps (Figure 3)](http://www.nanotomy.org/avivator/?image_url=http://www.nanotomy.org/OA/Duinkerken2024NPJI/figures/fig3/Figure3_Multichannel.ome.tif)

[Abundance map composite image (Figure 3)](http://www.nanotomy.org/avivator/?image_url=http://www.nanotomy.org/OA/Duinkerken2024NPJI/figures/fig3/Figure3_Multicolor.ome.tif)

[Skin HAADF and elemental maps (Figure S1)](http://www.nanotomy.org/avivator/?image_url=http://www.nanotomy.org/OA/Duinkerken2024NPJI/figures/figS1/FigureS1_Elements.ome.tif)

[Skin abundance maps (Figure S1)](http://www.nanotomy.org/avivator/?image_url=http://www.nanotomy.org/OA/Duinkerken2024NPJI/figures/figS1/FigureS1_AbundanceMaps.ome.tif)



Acknowledgements
---------
The endmember extraction algorithm used in this study was inspired by the work of [Vermeulen et al. (2021)](https://www.sciencedirect.com/science/article/abs/pii/S1386142521001232), and using code from the corresponding [repository](https://github.com/NU-ACCESS/UMAP). Implementation of SAM segmentation (Figure 4) uses code adapted from the following repositories: [SAM](https://github.com/facebookresearch/segment-anything), [SAMHQ](https://github.com/SysCV/sam-hq). VCA from this [repositoriy](https://github.com/Laadr/VCA) (used in preliminary testing; unused in paper).

Licensing
---------

Copyright (C) 2024 Ahmad Alsahaf and Peter Duinkerken.
