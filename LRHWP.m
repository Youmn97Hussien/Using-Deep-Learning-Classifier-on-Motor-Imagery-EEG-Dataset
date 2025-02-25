
PreSleepData = []; 
PosSleepData = []; 
preuuid='1';
posuuid= '1';
PreSleepFolderPath = 'D:\_temp_matlab_R2024b_Windows\matlab\LRHWakeClassifier\imagery_presleep';
PosSleepFolderPath = 'D:\_temp_matlab_R2024b_Windows\matlab\LRHWakeClassifier\imagery_postsleep'; 

PreSleepallFiles = dir(fullfile(PreSleepFolderPath, '*.mat')); 
PosSleepallFiles = dir(fullfile(PosSleepFolderPath, '*.mat')); 

for i = 1:length(PreSleepallFiles) 
    FilePath = fullfile(PreSleepFolderPath, PreSleepallFiles(i).name); 
    PreSleepData = load(FilePath);    
    PreprocessingData(FilePath,PreSleepData,preuuid)   
    preuuid = num2str(str2double(preuuid) + 1);
 end 
for j = 1:length(PosSleepallFiles) 
    FilePath = fullfile(PosSleepFolderPath, PosSleepallFiles(j).name); 
    PosSleepData = load(FilePath);    
    PreprocessingData(FilePath,PosSleepData,posuuid)   
    posuuid = num2str(str2double(posuuid) + 1);
 end 
function PreprocessingData(FilePath,SleepData,uuid)
    cfg = []; 
    cfg.channel = setdiff(SleepData.dat.label, {'EOG1', 'EOG2'}); 
    SleepData = ft_selectdata(cfg, SleepData.dat);
    
    cfg =[];
    cfg.toilim = [-0.2 0.9];% baseline needs more thinking 
    cfg.toilim = [0 1.1];
    SleepData = ft_redefinetrial(cfg, SleepData);
    cfg.baselinewindow  = [-0.2 0];
    
    cfg.bpfilter = 'yes'; 
    cfg.bpfreq = [0.1 40]; 
    cfg.bpfilttype = 'but';  
    cfg.bpfiltord = 4;
    
    cfg.reref = 'yes'; 
    cfg.refmethod = 'avg';
    cfg.refchannel = {'FZ', 'CZ', 'PZ', 'F3', 'F4', 'C5', 'CP3', 'C6', 'CP4', 'P7', 'P8', 'O1', 'O2'};
    
    FilteredData = ft_preprocessing(cfg, SleepData); 
    labels = [];    
    for i = 1:length(FilteredData.trialinfo) 
        labels = [labels; FilteredData.trialinfo{i, 1}.trialLabel];        
    end  
    [channels, time] = size(FilteredData.trial{1});
    features = zeros(length(FilteredData.trial),channels ,time );    
    for i = 1 :length(FilteredData.trial)
    features(i, :, :) = FilteredData.trial{i};    
    end   
    
    if contains(FilePath, 'D:\_temp_matlab_R2024b_Windows\matlab\LRHWakeClassifier\imagery_presleep') 
        labels_filename = sprintf('D:\\_temp_matlab_R2024b_Windows\\matlab\\LRHWakeClassifier\\imagery_presleep\\labels\\labels_%s.mat', uuid); 
        features_filename = sprintf('D:\\_temp_matlab_R2024b_Windows\\matlab\\LRHWakeClassifier\\imagery_presleep\\features\\features_%s.mat', uuid); 
        save(labels_filename, 'labels');
        save(features_filename, 'features');
        
    else         
        labels_filename = sprintf('D:\\_temp_matlab_R2024b_Windows\\matlab\\LRHWakeClassifier\\imagery_postsleep\\labels\\labels_%s.mat', uuid); 
        features_filename = sprintf('D:\\_temp_matlab_R2024b_Windows\\matlab\\LRHWakeClassifier\\imagery_postsleep\\features\\features_%s.mat', uuid); 
        save(labels_filename, 'labels'); 
        save(features_filename, 'features');        
    end  
        %
            %figure;
        %plot(SleepData.time{1}, SleepData.trial{1});
        %figure;
        %plot(FilteredData.time{1}, FilteredData.trial{1}); 
          %    
end