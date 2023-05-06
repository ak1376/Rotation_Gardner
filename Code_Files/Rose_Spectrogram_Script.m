%% This function uses SAT_sound to calculate properties of the input audio file, then calculates the temporal derivative

function out = derivSpecGram(audioFileName, smoothingAmt)
if isempty(smoothingAmt)
    smoothingAmt = 17.64;
end
% SAT_sound to calculate the sound features:
sound=SAT_sound(audioFileName, 0); % do not plot results
out.sound = sound;
sononogramData = sound.sonogram;
absAudioSpec = abs(sononogramData);
spectrogramLengthVec = 1:length(sononogramData);
[FX, FY] = gradient(sononogramData); %find the derivative
summedGradient = sum(FX, 1); %take the sum of the derivative for each column
%smooth the derivative with a Gaussian function: 
smoothedGradient = smoothdata(summedGradient,'gaussian', smoothingAmt); %smooth with 0.41 ms Gaussian (100/44100)  = 17.64 :)

%make a subplot of the spectrogram:
subplot(2,1,1);
imagesc(flipdim(log(absAudioSpec+0.00001), 1));
xlabel(gca, 'Time (ms)');
ylabel(gca, 'Frequency (Hz)');
title(gca, 'Spectrogram of audio file');

%Make a subplot of the power derivative:
subplot(2,1,2);
plot(spectrogramLengthVec, smoothedGradient);
xlabel(gca, 'Time (ms)');
ylabel(gca, 'derivative of song power');
title(gca, 'Temporal derivative of song power');
xlim(gca, [1 length(summedGradient)]);
end %function's end