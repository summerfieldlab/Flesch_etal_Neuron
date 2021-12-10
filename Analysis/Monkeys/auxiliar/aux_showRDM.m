function aux_showRDM(rdm)

    image(scale01(rankTransform_equalsStayEqual(rdm)),'CDataMapping','scaled','AlphaData',1);
    colormap('parula');

end