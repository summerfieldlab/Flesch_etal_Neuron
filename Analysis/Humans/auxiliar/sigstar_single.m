function sigstar_single(xPos,yPos,p)

if p<=1E-3
   stars='***';
elseif p<=1E-2
   stars='**';
elseif p<=0.05
   stars='*';
elseif isnan(p)
   stars='n.s.';
else
   stars='n.s.'; % orig: ''
end
if ~isnan(p)
    offset=0.08; % orig: 0.005
else
    offset=0.08;  % orig: 0.02
end


text(xPos,yPos+yPos*offset,stars,'HorizontalAlignment','Center','BackGroundColor','none','Tag','sigstars');

end
