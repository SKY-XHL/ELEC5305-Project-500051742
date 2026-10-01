figure('Color','w','Position',[100 100 900 700]);
axis off
hold on

% ===== Main vertical boxes =====
boxW = 0.22;
boxH = 0.06;
xMain = 0.38;

y1 = 0.88;
y2 = 0.78;
y3 = 0.68;
y4 = 0.58;
y5 = 0.48;
y6 = 0.35;
y7 = 0.25;
y8 = 0.15;

drawBox(xMain,y1,boxW,boxH,'Audio Files');
drawBox(xMain,y2,boxW,boxH,'File Check and Loading');
drawBox(xMain,y3,boxW,boxH,'Audio Property Inspection');
drawBox(xMain,y4,boxW,boxH,'Time-Domain Waveform Analysis');
drawBox(xMain,y5,boxW,boxH,'Frequency-Domain FFT Analysis');
drawBox(xMain,y6,boxW,boxH,'STFT Spectrogram Analysis');
drawBox(xMain,y7,boxW,boxH,'Window-Length Comparison');
drawBox(xMain,y8,boxW,boxH,'Result Interpretation');

% ===== Arrows in main flow =====
drawArrow(xMain+boxW/2, y1, xMain+boxW/2, y2+boxH);
drawArrow(xMain+boxW/2, y2, xMain+boxW/2, y3+boxH);
drawArrow(xMain+boxW/2, y3, xMain+boxW/2, y4+boxH);
drawArrow(xMain+boxW/2, y4, xMain+boxW/2, y5+boxH);
drawArrow(xMain+boxW/2, y5, xMain+boxW/2, y6+boxH);
drawArrow(xMain+boxW/2, y6, xMain+boxW/2, y7+boxH);
drawArrow(xMain+boxW/2, y7, xMain+boxW/2, y8+boxH);

% ===== Side boxes for STFT details =====
boxW2 = 0.20;
boxH2 = 0.055;
xRight = 0.72;

yr1 = 0.40;
yr2 = 0.32;
yr3 = 0.24;

drawBox(xRight,yr1,boxW2,boxH2,'Short Window STFT');
drawBox(xRight,yr2,boxW2,boxH2,'Medium Window STFT');
drawBox(xRight,yr3,boxW2,boxH2,'Long Window STFT');

% arrows from STFT box to side boxes
drawArrow(xMain+boxW, y6+boxH/2, xRight, yr1+boxH2/2);
drawArrow(xMain+boxW, y6+boxH/2, xRight, yr2+boxH2/2);
drawArrow(xMain+boxW, y6+boxH/2, xRight, yr3+boxH2/2);

% ===== Caption =====
annotation('textbox',[0.22 0.02 0.6 0.05], ...
    'String','Figure 1. Block diagram of the audio signal processing pipeline.', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontWeight','bold','FontSize',12);

% ===== Export =====
exportgraphics(gcf,'processing_pipeline.png','Resolution',300);

% ===== Helper functions =====
function drawBox(x,y,w,h,str)
    annotation('textbox',[x y w h], ...
        'String',str, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontSize',11, ...
        'EdgeColor',[0.4 0.4 0.4], ...
        'LineWidth',1, ...
        'BackgroundColor',[0.90 0.94 0.98], ...
        'FitBoxToText','off');
end

function drawArrow(x1,y1,x2,y2)
    annotation('arrow',[x1 x2],[y1 y2], ...
        'Color',[0.3 0.3 0.3], ...
        'LineWidth',1);
end